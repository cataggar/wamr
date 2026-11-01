const std = @import("std");
const json = std.json;
const Allocator = std.mem.Allocator;
const Value = json.Value;
const ObjectMap = json.ObjectMap;
const Array = json.Array;

pub const SCHEMA_VERSION: i64 = 2;
pub const ANALYSIS_KIND = "aarch64-conservative-alu-provenance";

pub const CATEGORIES = [_][]const u8{
    "address_generation",
    "structural_address_guard",
    "algorithmic_alu",
    "mixed",
    "unknown",
};

pub const GATE_CATEGORIES = [_][]const u8{
    "address_generation",
    "algorithmic_alu",
};

pub const PROVEN_CATEGORIES = [_][]const u8{
    "address_generation",
    "structural_address_guard",
    "algorithmic_alu",
};

pub const DEFAULT_GATE_THRESHOLD_PCT: f64 = 5.0;

pub const UNKNOWN_REASON_CLUSTERS = [_][]const u8{
    "non_trap_control",
    "ambiguous_cfg_join",
    "cyclic_flow",
    "untyped_escape",
    "opaque_or_call_boundary",
    "combined_memory_writeback",
    "dead_or_unconsumed",
    "mixed_categories",
};

pub const TERMINAL_SINK_CLASSES = [_][]const u8{
    "pure_address",
    "pure_structural_guard",
    "pure_algorithmic",
    "non_trap_control",
    "untyped_escape",
    "mixed_address_and_control_or_escape",
    "opaque_or_call_boundary",
    "combined_memory_writeback",
    "dead_or_unconsumed",
};

pub const TRACKED_REGISTERS = [_][]const u8{
    "x0",   "x1",  "x2",  "x3",  "x4",  "x5",  "x6",  "x7",
    "x8",   "x9",  "x10", "x11", "x12", "x13", "x14", "x15",
    "x16",  "x17", "x18", "x19", "x20", "x21", "x22", "x23",
    "x24",  "x25", "x26", "x27", "x28", "x29", "x30", "sp",
    "nzcv",
};

pub const CALLER_CLOBBERED = [_][]const u8{
    "x0",  "x1",  "x2",  "x3",  "x4",   "x5",  "x6",  "x7",
    "x8",  "x9",  "x10", "x11", "x12",  "x13", "x14", "x15",
    "x16", "x17", "x18", "x30", "nzcv",
};

pub const POSSIBLE_CALL_ARGUMENTS = [_][]const u8{
    "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7",
};

pub const POSSIBLE_RETURN_VALUES = [_][]const u8{ "x0", "x1" };

const LOAD_PREFIXES = [_][]const u8{
    "ldr",
    "ldur",
    "ldp",
    "ldnp",
    "ldtr",
    "ldxr",
    "ldaxr",
    "ldar",
    "ldapr",
};

const STORE_PREFIXES = [_][]const u8{
    "str",
    "stur",
    "stp",
    "stnp",
    "sttr",
    "stxr",
    "stlxr",
    "stlr",
};

const ALU_DEST_MNEMONICS = [_][]const u8{
    "add",   "adds",  "sub",    "subs",   "adc",    "adcs",
    "sbc",   "sbcs",  "and",    "ands",   "orr",    "orn",
    "eor",   "eon",   "bic",    "bics",   "madd",   "msub",
    "mul",   "mneg",  "smaddl", "smsubl", "umaddl", "umsubl",
    "smulh", "umulh", "smull",  "umull",  "udiv",   "sdiv",
    "lsl",   "lsr",   "asr",    "ror",    "extr",   "ubfm",
    "sbfm",  "bfm",   "ubfx",   "sbfx",   "bfi",    "bfxil",
    "uxtb",  "uxth",  "uxtw",   "sxtb",   "sxth",   "sxtw",
    "csel",  "csinc", "csinv",  "csneg",  "cset",   "csetm",
    "neg",   "negs",  "mvn",    "adr",    "adrp",
};

const FLAG_ONLY_MNEMONICS = [_][]const u8{ "cmp", "cmn", "tst" };
const CONDITIONAL_FLAG_MNEMONICS = [_][]const u8{ "ccmp", "ccmn" };
const FLOAT_FLAG_MNEMONICS = [_][]const u8{ "fcmp", "fcmpe", "fccmp", "fccmpe" };
const FLAG_WRITING_MNEMONICS = [_][]const u8{
    "adds", "subs", "adcs", "sbcs", "ands", "bics", "negs",
};
const MOVE_MNEMONICS = [_][]const u8{ "mov", "movz", "movn", "movk", "fmov" };
const VALUE_TRANSFORM_MNEMONICS = [_][]const u8{
    "rev", "rev16", "rev32", "rbit", "clz", "cls",
};
const TRAP_MNEMONICS = [_][]const u8{ "brk", "hlt", "udf" };
const SIDE_EFFECT_FREE_TRAP_PATH_MNEMONICS = [_][]const u8{"nop"};

pub const ProvenanceError = error{
    ProvenanceFailed,
    OutOfMemory,
};

pub const Diagnostic = struct {
    message: []const u8 = "",

    pub fn fail(self: *Diagnostic, allocator: Allocator, comptime fmt: []const u8, args: anytype) ProvenanceError {
        self.message = try std.fmt.allocPrint(allocator, fmt, args);
        return error.ProvenanceFailed;
    }
};

pub const InstructionInput = struct {
    offset: i64,
    size: i64,
    mnemonic: []const u8 = "",
    operands: []const u8 = "",
    text: []const u8,
    address: ?i64 = null,
};

pub const Instruction = struct {
    offset: i64,
    size: i64,
    mnemonic: []const u8,
    operands: []const u8,
    text: []const u8,
    address: i64,
};

pub const Use = struct {
    register: []const u8,
    role: []const u8,
    propagate_to: []const []const u8 = &.{},
};

pub const Effect = struct {
    uses: std.ArrayList(Use) = .empty,
    definitions: std.StringArrayHashMapUnmanaged(void) = .empty,
    primary_definitions: std.StringArrayHashMapUnmanaged(void) = .empty,
    secondary_definitions: std.StringArrayHashMapUnmanaged(void) = .empty,
    branch_kind: ?[]const u8 = null,
    branch_target: ?i64 = null,
    trap: bool = false,
    memory: bool = false,
    recognized: bool = true,
    gating_candidate: bool = false,
    opaque_barrier: bool = false,
};

pub const Definition = struct {
    instruction: i64,
    register: []const u8,

    pub fn eql(a: Definition, b: Definition) bool {
        return a.instruction == b.instruction and std.mem.eql(u8, a.register, b.register);
    }
};

const DefinitionContext = struct {
    pub fn hash(_: DefinitionContext, d: Definition) u32 {
        var h = std.hash.Wyhash.init(0);
        h.update(std.mem.asBytes(&d.instruction));
        h.update(d.register);
        return @truncate(h.final());
    }

    pub fn eql(_: DefinitionContext, a: Definition, b: Definition, _: usize) bool {
        return Definition.eql(a, b);
    }
};

const DefinitionSet = std.ArrayHashMapUnmanaged(Definition, void, DefinitionContext, true);

fn defSetEqual(a: DefinitionSet, b: DefinitionSet) bool {
    if (a.count() != b.count()) return false;
    var it = a.iterator();
    while (it.next()) |entry| {
        if (!b.contains(entry.key_ptr.*)) return false;
    }
    return true;
}

fn defSetClone(allocator: Allocator, src: DefinitionSet) !DefinitionSet {
    var out: DefinitionSet = .empty;
    var it = src.iterator();
    while (it.next()) |entry| {
        try out.put(allocator, entry.key_ptr.*, {});
    }
    return out;
}

pub const Consumer = struct {
    instruction: usize,
    use: Use,
    ambiguous: bool,
};

pub const EvidenceItem = struct {
    offset: i64,
    instruction: []const u8,
    role: []const u8,
};

pub const Trace = struct {
    roles: std.StringArrayHashMapUnmanaged(void) = .empty,
    unknown_reasons: std.StringArrayHashMapUnmanaged(void) = .empty,
    evidence: std.ArrayList(EvidenceItem) = .empty,

    pub fn clone(self: Trace, allocator: Allocator) !Trace {
        var out: Trace = .{};
        var r_it = self.roles.iterator();
        while (r_it.next()) |entry| try out.roles.put(allocator, entry.key_ptr.*, {});
        var u_it = self.unknown_reasons.iterator();
        while (u_it.next()) |entry| try out.unknown_reasons.put(allocator, entry.key_ptr.*, {});
        try out.evidence.appendSlice(allocator, self.evidence.items);
        return out;
    }

    pub fn merge(self: *Trace, allocator: Allocator, other: Trace) !void {
        var r_it = other.roles.iterator();
        while (r_it.next()) |entry| try self.roles.put(allocator, entry.key_ptr.*, {});
        var u_it = other.unknown_reasons.iterator();
        while (u_it.next()) |entry| try self.unknown_reasons.put(allocator, entry.key_ptr.*, {});
        try self.evidence.appendSlice(allocator, other.evidence.items);
    }
};

fn inList(haystack: []const []const u8, needle: []const u8) bool {
    for (haystack) |item| {
        if (std.mem.eql(u8, item, needle)) return true;
    }
    return false;
}

fn hasPrefixInList(prefixes: []const []const u8, text: []const u8) bool {
    for (prefixes) |p| {
        if (std.mem.startsWith(u8, text, p)) return true;
    }
    return false;
}

fn asciiLowerDup(allocator: Allocator, text: []const u8) ![]u8 {
    const out = try allocator.dupe(u8, text);
    for (out) |*c| c.* = std.ascii.toLower(c.*);
    return out;
}

pub fn splitOperands(allocator: Allocator, text: []const u8) ![]const []const u8 {
    var result: std.ArrayList([]const u8) = .empty;
    var current: std.ArrayList(u8) = .empty;
    var depth: usize = 0;
    for (text) |ch| {
        if (ch == '[' or ch == '(') {
            depth += 1;
        } else if (ch == ']' or ch == ')') {
            if (depth > 0) depth -= 1;
        }
        if (ch == ',' and depth == 0) {
            const trimmed = std.mem.trim(u8, current.items, " \t\r\n");
            if (trimmed.len > 0) {
                try result.append(allocator, try allocator.dupe(u8, trimmed));
            }
            current.clearRetainingCapacity();
        } else {
            try current.append(allocator, ch);
        }
    }
    if (current.items.len > 0 or std.mem.trim(u8, text, " \t\r\n").len > 0) {
        const trimmed = std.mem.trim(u8, current.items, " \t\r\n");
        if (trimmed.len > 0) {
            try result.append(allocator, try allocator.dupe(u8, trimmed));
        }
    }
    return result.toOwnedSlice(allocator);
}

pub fn splitInstructionText(allocator: Allocator, text: []const u8) !struct { mnemonic: []const u8, operands: []const u8 } {
    const trimmed = std.mem.trim(u8, text, " \t\r\n");
    if (trimmed.len == 0) return .{ .mnemonic = "", .operands = "" };
    var idx: usize = 0;
    while (idx < trimmed.len and !std.ascii.isWhitespace(trimmed[idx])) : (idx += 1) {}
    const mnemonic = try asciiLowerDup(allocator, trimmed[0..idx]);
    const rest = if (idx < trimmed.len) std.mem.trim(u8, trimmed[idx..], " \t\r\n") else "";
    return .{ .mnemonic = mnemonic, .operands = try allocator.dupe(u8, rest) };
}

pub fn canonicalRegister(value: []const u8) ?[]const u8 {
    var buf: [8]u8 = undefined;
    if (value.len == 0 or value.len > buf.len) return null;
    for (value, 0..) |ch, i| buf[i] = std.ascii.toLower(ch);
    const register = buf[0..value.len];

    if (std.mem.eql(u8, register, "xzr") or std.mem.eql(u8, register, "wzr")) return null;
    if (std.mem.eql(u8, register, "fp")) return "x29";
    if (std.mem.eql(u8, register, "lr")) return "x30";
    if (std.mem.eql(u8, register, "sp") or std.mem.eql(u8, register, "wsp")) return "sp";
    if ((register[0] == 'w' or register[0] == 'x') and register.len >= 2) {
        const digits = register[1..];
        for (digits) |d| {
            if (!std.ascii.isDigit(d)) return null;
        }
        const index = std.fmt.parseInt(usize, digits, 10) catch return null;
        if (index <= 30) {
            return TRACKED_REGISTERS[index];
        }
    }
    return null;
}

fn isIdentChar(ch: u8) bool {
    return std.ascii.isAlphanumeric(ch) or ch == '_';
}

pub fn registersInOperand(allocator: Allocator, operand: []const u8) ![]const []const u8 {
    var result: std.ArrayList([]const u8) = .empty;
    var i: usize = 0;
    while (i < operand.len) {
        if (!isIdentChar(operand[i])) {
            i += 1;
            continue;
        }
        const start = i;
        while (i < operand.len and isIdentChar(operand[i])) : (i += 1) {}
        const token = operand[start..i];
        if (canonicalRegister(token)) |canon| {
            if (!inList(result.items, canon)) {
                try result.append(allocator, canon);
            }
        }
    }
    return result.toOwnedSlice(allocator);
}

pub fn normalizeInstructions(
    allocator: Allocator,
    instructions: []const InstructionInput,
    address_base: i64,
    diag: *Diagnostic,
) ProvenanceError![]Instruction {
    if (instructions.len == 0) {
        return diag.fail(allocator, "disassembly contained no instructions", .{});
    }
    var normalized: std.ArrayList(Instruction) = .empty;
    var previous: i64 = -1;
    for (instructions) |item| {
        const trimmed_text = std.mem.trim(u8, item.text, " \t\r\n");
        if (item.offset <= previous or item.size <= 0 or trimmed_text.len == 0) {
            return diag.fail(
                allocator,
                "disassembly instructions must have increasing offsets, positive sizes, and nonempty text",
                .{},
            );
        }
        var mnemonic = try asciiLowerDup(allocator, item.mnemonic);
        var operands = try allocator.dupe(u8, item.operands);
        if (mnemonic.len == 0) {
            const split = try splitInstructionText(allocator, trimmed_text);
            mnemonic = @constCast(split.mnemonic);
            operands = @constCast(split.operands);
        }
        const address = item.address orelse (address_base + item.offset);
        try normalized.append(allocator, .{
            .offset = item.offset,
            .size = item.size,
            .mnemonic = mnemonic,
            .operands = operands,
            .text = try allocator.dupe(u8, trimmed_text),
            .address = address,
        });
        previous = item.offset;
    }
    return normalized.toOwnedSlice(allocator);
}

fn addUses(
    allocator: Allocator,
    effect: *Effect,
    operands: []const []const u8,
    role: []const u8,
    propagate_to: []const []const u8,
) !void {
    for (operands) |operand| {
        const regs = try registersInOperand(allocator, operand);
        for (regs) |reg| {
            try effect.uses.append(allocator, .{
                .register = reg,
                .role = role,
                .propagate_to = propagate_to,
            });
        }
    }
}

fn parseTarget(operand: []const u8) ?i64 {
    var cleaned_buf: [128]u8 = undefined;
    var c_len: usize = 0;
    for (operand) |ch| {
        if (ch != '#' and c_len < cleaned_buf.len) {
            cleaned_buf[c_len] = ch;
            c_len += 1;
        }
    }
    const text = cleaned_buf[0..c_len];
    var i: usize = 0;
    while (i < text.len) : (i += 1) {
        const ch = text[i];
        const has_sign = (ch == '+' or ch == '-') and (i + 1 < text.len and std.ascii.isDigit(text[i + 1]));
        if (!has_sign and !std.ascii.isDigit(ch)) continue;

        const start = i;
        if (has_sign) i += 1;
        if (i + 1 < text.len and text[i] == '0' and (text[i + 1] == 'x' or text[i + 1] == 'X')) {
            i += 2;
            const hex_start = i;
            while (i < text.len and std.ascii.isHex(text[i])) : (i += 1) {}
            if (i > hex_start) {
                return std.fmt.parseInt(i64, text[start..i], 0) catch null;
            }
        } else {
            while (i < text.len and std.ascii.isDigit(text[i])) : (i += 1) {}
            return std.fmt.parseInt(i64, text[start..i], 10) catch null;
        }
    }
    return null;
}

fn isLoad(mnemonic: []const u8) bool {
    return hasPrefixInList(&LOAD_PREFIXES, mnemonic);
}

fn isStore(mnemonic: []const u8) bool {
    return hasPrefixInList(&STORE_PREFIXES, mnemonic);
}

fn opaqueEffect(allocator: Allocator, gating_candidate: bool) !Effect {
    var effect: Effect = .{
        .recognized = false,
        .gating_candidate = gating_candidate,
        .opaque_barrier = true,
        .branch_kind = "opaque",
    };
    for (TRACKED_REGISTERS) |reg| {
        try effect.uses.append(allocator, .{ .register = reg, .role = "unknown" });
        try effect.definitions.put(allocator, reg, {});
        try effect.secondary_definitions.put(allocator, reg, {});
    }
    return effect;
}

fn memoryEffect(allocator: Allocator, instruction: Instruction, operands: []const []const u8) !Effect {
    var effect: Effect = .{ .memory = true };
    var memory_indices: std.ArrayList(usize) = .empty;
    for (operands, 0..) |op, idx| {
        if (std.mem.indexOfScalar(u8, op, '[') != null) {
            try memory_indices.append(allocator, idx);
        }
    }
    if (memory_indices.items.len == 0 and isLoad(instruction.mnemonic) and operands.len >= 2) {
        const destination = try registersInOperand(allocator, operands[0]);
        if (destination.len == 1) {
            try effect.definitions.put(allocator, destination[0], {});
            try effect.primary_definitions.put(allocator, destination[0], {});
        }
        return effect;
    }
    if (memory_indices.items.len != 1) {
        return opaqueEffect(allocator, false);
    }
    const memory_index = memory_indices.items[0];
    const memory_operand = operands[memory_index];
    const address_registers = try registersInOperand(allocator, memory_operand);
    for (address_registers) |reg| {
        try effect.uses.append(allocator, .{ .register = reg, .role = "address" });
    }

    const load = isLoad(instruction.mnemonic);
    const store = isStore(instruction.mnemonic);
    if (!load and !store) {
        return opaqueEffect(allocator, false);
    }

    var value_operands = operands[0..memory_index];
    if (load) {
        for (value_operands) |op| {
            const regs = try registersInOperand(allocator, op);
            if (regs.len == 1) {
                try effect.definitions.put(allocator, regs[0], {});
                try effect.primary_definitions.put(allocator, regs[0], {});
            }
        }
    } else {
        const status_result = std.mem.startsWith(u8, instruction.mnemonic, "stxr") or
            std.mem.startsWith(u8, instruction.mnemonic, "stlxr");
        if (status_result and value_operands.len > 0) {
            const regs = try registersInOperand(allocator, value_operands[0]);
            if (regs.len == 1) {
                try effect.definitions.put(allocator, regs[0], {});
                try effect.primary_definitions.put(allocator, regs[0], {});
            } else {
                effect.recognized = false;
            }
            value_operands = value_operands[1..];
        }
        try addUses(allocator, &effect, value_operands, "escape", &.{});
    }

    var writeback = std.mem.endsWith(u8, std.mem.trimEnd(u8, memory_operand, " \t\r\n"), "]!");
    const post_index_operands = operands[memory_index + 1 ..];
    if (post_index_operands.len > 0) {
        writeback = true;
        try addUses(allocator, &effect, post_index_operands, "value", &.{});
    }
    if (writeback) {
        effect.gating_candidate = true;
        if (address_registers.len == 0) {
            return opaqueEffect(allocator, false);
        }
        const base = address_registers[0];
        try effect.definitions.put(allocator, base, {});
        try effect.secondary_definitions.put(allocator, base, {});
        const prop = try allocator.dupe([]const u8, &.{base});
        for (effect.uses.items) |*u| {
            if (std.mem.eql(u8, u.register, base) and std.mem.eql(u8, u.role, "address")) {
                u.propagate_to = prop;
            } else {
                u.propagate_to = &.{};
            }
        }
    }
    return effect;
}

pub fn instructionEffect(allocator: Allocator, instruction: Instruction) !Effect {
    const mnemonic = instruction.mnemonic;
    const operands = try splitOperands(allocator, instruction.operands);

    if (isLoad(mnemonic) or isStore(mnemonic)) {
        return memoryEffect(allocator, instruction, operands);
    }
    if (std.mem.eql(u8, mnemonic, "prfm")) {
        var mem_ops: std.ArrayList([]const u8) = .empty;
        for (operands) |op| {
            if (std.mem.indexOfScalar(u8, op, '[') != null) {
                try mem_ops.append(allocator, op);
            }
        }
        if (mem_ops.items.len != 1) {
            return opaqueEffect(allocator, false);
        }
        var effect: Effect = .{ .memory = true };
        try addUses(allocator, &effect, mem_ops.items, "address", &.{});
        return effect;
    }

    var effect: Effect = .{};
    if (inList(&TRAP_MNEMONICS, mnemonic)) {
        effect.trap = true;
        effect.branch_kind = "trap";
        return effect;
    }
    if (inList(&SIDE_EFFECT_FREE_TRAP_PATH_MNEMONICS, mnemonic)) {
        return effect;
    }
    if (std.mem.eql(u8, mnemonic, "b")) {
        effect.branch_kind = "unconditional";
        effect.branch_target = if (operands.len > 0) parseTarget(operands[operands.len - 1]) else null;
        return effect;
    }
    if (std.mem.startsWith(u8, mnemonic, "b.")) {
        effect.branch_kind = "conditional";
        effect.branch_target = if (operands.len > 0) parseTarget(operands[operands.len - 1]) else null;
        try effect.uses.append(allocator, .{ .register = "nzcv", .role = "control" });
        return effect;
    }
    if (std.mem.eql(u8, mnemonic, "cbz") or
        std.mem.eql(u8, mnemonic, "cbnz") or
        std.mem.eql(u8, mnemonic, "tbz") or
        std.mem.eql(u8, mnemonic, "tbnz"))
    {
        effect.branch_kind = "conditional";
        effect.branch_target = if (operands.len > 0) parseTarget(operands[operands.len - 1]) else null;
        if (operands.len > 0) {
            try addUses(allocator, &effect, operands[0..1], "control", &.{});
        }
        return effect;
    }
    if (std.mem.eql(u8, mnemonic, "br")) {
        effect.branch_kind = "indirect";
        if (operands.len > 0) {
            try addUses(allocator, &effect, operands[0..1], "control", &.{});
        }
        for (TRACKED_REGISTERS) |reg| {
            if (!std.mem.eql(u8, reg, "sp") and !std.mem.eql(u8, reg, "nzcv")) {
                try effect.uses.append(allocator, .{ .register = reg, .role = "unknown" });
            }
        }
        return effect;
    }
    if (std.mem.eql(u8, mnemonic, "bl") or std.mem.eql(u8, mnemonic, "blr")) {
        effect.branch_kind = "call";
        if (std.mem.eql(u8, mnemonic, "blr") and operands.len > 0) {
            try addUses(allocator, &effect, operands[0..1], "control", &.{});
        }
        for (POSSIBLE_CALL_ARGUMENTS) |reg| {
            try effect.uses.append(allocator, .{ .register = reg, .role = "unknown" });
        }
        for (CALLER_CLOBBERED) |reg| {
            try effect.definitions.put(allocator, reg, {});
            try effect.secondary_definitions.put(allocator, reg, {});
        }
        return effect;
    }
    if (std.mem.eql(u8, mnemonic, "ret")) {
        effect.branch_kind = "return";
        if (operands.len > 0) {
            try addUses(allocator, &effect, operands[0..1], "control", &.{});
        } else {
            try effect.uses.append(allocator, .{ .register = "x30", .role = "control" });
        }
        for (POSSIBLE_RETURN_VALUES) |reg| {
            try effect.uses.append(allocator, .{ .register = reg, .role = "escape" });
        }
        return effect;
    }

    if (inList(&CONDITIONAL_FLAG_MNEMONICS, mnemonic)) {
        effect.gating_candidate = true;
        try effect.definitions.put(allocator, "nzcv", {});
        try effect.primary_definitions.put(allocator, "nzcv", {});
        try effect.uses.append(allocator, .{ .register = "nzcv", .role = "unknown" });
        try addUses(allocator, &effect, operands, "unknown", &.{});
        return effect;
    }

    if (inList(&FLAG_ONLY_MNEMONICS, mnemonic)) {
        effect.gating_candidate = true;
        try effect.definitions.put(allocator, "nzcv", {});
        try effect.primary_definitions.put(allocator, "nzcv", {});
        try addUses(allocator, &effect, operands, "compare", &.{});
        return effect;
    }

    if (inList(&FLOAT_FLAG_MNEMONICS, mnemonic)) {
        effect.gating_candidate = true;
        try effect.definitions.put(allocator, "nzcv", {});
        try effect.primary_definitions.put(allocator, "nzcv", {});
        if (std.mem.startsWith(u8, mnemonic, "fcc")) {
            try effect.uses.append(allocator, .{ .register = "nzcv", .role = "unknown" });
        }
        return effect;
    }

    if (inList(&ALU_DEST_MNEMONICS, mnemonic) or inList(&VALUE_TRANSFORM_MNEMONICS, mnemonic)) {
        effect.gating_candidate = true;
        if (operands.len == 0) return opaqueEffect(allocator, true);
        const destination = try registersInOperand(allocator, operands[0]);
        if (destination.len != 1) return opaqueEffect(allocator, true);
        const dest = destination[0];
        try effect.definitions.put(allocator, dest, {});
        try effect.primary_definitions.put(allocator, dest, {});
        var propagated: std.ArrayList([]const u8) = .empty;
        try propagated.append(allocator, dest);
        if (inList(&FLAG_WRITING_MNEMONICS, mnemonic)) {
            try effect.definitions.put(allocator, "nzcv", {});
            try effect.secondary_definitions.put(allocator, "nzcv", {});
            try propagated.append(allocator, "nzcv");
        }
        const prop_slice = try propagated.toOwnedSlice(allocator);
        const sources = operands[1..];
        if (std.mem.eql(u8, mnemonic, "cset") or std.mem.eql(u8, mnemonic, "csetm")) {
            const dest_only = try allocator.dupe([]const u8, &.{dest});
            try effect.uses.append(allocator, .{ .register = "nzcv", .role = "value", .propagate_to = dest_only });
        } else {
            try addUses(allocator, &effect, sources, "value", prop_slice);
            if (std.mem.startsWith(u8, mnemonic, "cs")) {
                const dest_only = try allocator.dupe([]const u8, &.{dest});
                try effect.uses.append(allocator, .{ .register = "nzcv", .role = "value", .propagate_to = dest_only });
            }
        }
        return effect;
    }

    if (inList(&MOVE_MNEMONICS, mnemonic)) {
        effect.gating_candidate = true;
        if (operands.len == 0) return opaqueEffect(allocator, true);
        const destination = try registersInOperand(allocator, operands[0]);
        if (destination.len != 1) return opaqueEffect(allocator, true);
        const dest = destination[0];
        try effect.definitions.put(allocator, dest, {});
        try effect.primary_definitions.put(allocator, dest, {});
        const dest_only = try allocator.dupe([]const u8, &.{dest});
        const sources = operands[1..];
        if (std.mem.eql(u8, mnemonic, "movk")) {
            try effect.uses.append(allocator, .{ .register = dest, .role = "value", .propagate_to = dest_only });
        }
        try addUses(allocator, &effect, sources, "value", dest_only);
        return effect;
    }

    return opaqueEffect(allocator, true);
}

const IndexSet = std.AutoArrayHashMapUnmanaged(usize, void);

fn indexSetEqual(a: IndexSet, b: IndexSet) bool {
    if (a.count() != b.count()) return false;
    var it = a.iterator();
    while (it.next()) |entry| {
        if (!b.contains(entry.key_ptr.*)) return false;
    }
    return true;
}

pub const ExternalTarget = struct {
    offset: i64,
    instruction: []const u8,
    target: i64,
};

fn targetIndex(instructions: []const Instruction, target: ?i64) ?usize {
    const tgt = target orelse return null;
    for (instructions, 0..) |inst, idx| {
        if (inst.address == tgt) return idx;
    }
    return null;
}

pub fn buildCfg(
    allocator: Allocator,
    instructions: []const Instruction,
    effects: []Effect,
) !struct {
    successors: []IndexSet,
    predecessors: []IndexSet,
    external_targets: []ExternalTarget,
} {
    const successors = try allocator.alloc(IndexSet, instructions.len);
    const predecessors = try allocator.alloc(IndexSet, instructions.len);
    for (0..instructions.len) |i| {
        successors[i] = .empty;
        predecessors[i] = .empty;
    }
    var external_targets: std.ArrayList(ExternalTarget) = .empty;

    for (effects, 0..) |*effect, index| {
        const following: ?usize = if (index + 1 < instructions.len) index + 1 else null;
        const tgt_idx = targetIndex(instructions, effect.branch_target);
        if (effect.branch_target != null and tgt_idx == null) {
            try external_targets.append(allocator, .{
                .offset = instructions[index].offset,
                .instruction = instructions[index].text,
                .target = effect.branch_target.?,
            });
            if (effect.branch_kind != null and std.mem.eql(u8, effect.branch_kind.?, "unconditional")) {
                for (POSSIBLE_CALL_ARGUMENTS) |reg| {
                    try effect.uses.append(allocator, .{ .register = reg, .role = "unknown" });
                }
            }
        }
        const bk = effect.branch_kind;
        if (bk != null and std.mem.eql(u8, bk.?, "unconditional")) {
            if (tgt_idx) |ti| try successors[index].put(allocator, ti, {});
        } else if (bk != null and std.mem.eql(u8, bk.?, "conditional")) {
            if (following) |f| try successors[index].put(allocator, f, {});
            if (tgt_idx) |ti| try successors[index].put(allocator, ti, {});
        } else if (bk != null and (std.mem.eql(u8, bk.?, "indirect") or
            std.mem.eql(u8, bk.?, "return") or
            std.mem.eql(u8, bk.?, "trap") or
            std.mem.eql(u8, bk.?, "opaque")))
        {
            // no fallthrough
        } else if (following) |f| {
            try successors[index].put(allocator, f, {});
        }
    }

    for (successors, 0..) |targets, index| {
        var it = targets.iterator();
        while (it.next()) |entry| {
            try predecessors[entry.key_ptr.*].put(allocator, index, {});
        }
    }

    return .{
        .successors = successors,
        .predecessors = predecessors,
        .external_targets = try external_targets.toOwnedSlice(allocator),
    };
}

fn computeReachable(allocator: Allocator, successors: []const IndexSet) !IndexSet {
    var result: IndexSet = .empty;
    if (successors.len == 0) return result;
    var stack: std.ArrayList(usize) = .empty;
    try stack.append(allocator, 0);
    while (stack.items.len > 0) {
        const current = stack.pop().?;
        if (result.contains(current)) continue;
        try result.put(allocator, current, {});
        var it = successors[current].iterator();
        while (it.next()) |entry| {
            if (!result.contains(entry.key_ptr.*)) {
                try stack.append(allocator, entry.key_ptr.*);
            }
        }
    }
    return result;
}

const RegDefMap = std.StringArrayHashMapUnmanaged(DefinitionSet);

fn regDefMapEqual(a: RegDefMap, b: RegDefMap) bool {
    if (a.count() != b.count()) return false;
    var it = a.iterator();
    while (it.next()) |entry| {
        const b_set = b.get(entry.key_ptr.*) orelse return false;
        if (!defSetEqual(entry.value_ptr.*, b_set)) return false;
    }
    return true;
}

fn reachingDefinitions(
    allocator: Allocator,
    effects: []const Effect,
    predecessors: []const IndexSet,
    reachable: IndexSet,
) !struct { incoming: []RegDefMap, outgoing: []RegDefMap } {
    var entry_map: RegDefMap = .empty;
    for (TRACKED_REGISTERS) |reg| {
        var ds: DefinitionSet = .empty;
        try ds.put(allocator, .{ .instruction = -1, .register = reg }, {});
        try entry_map.put(allocator, reg, ds);
    }

    const incoming = try allocator.alloc(RegDefMap, effects.len);
    const outgoing = try allocator.alloc(RegDefMap, effects.len);
    for (0..effects.len) |i| {
        incoming[i] = .empty;
        outgoing[i] = .empty;
    }

    var changed = true;
    while (changed) {
        changed = false;
        for (effects, 0..) |effect, index| {
            if (!reachable.contains(index)) continue;
            var next_incoming: RegDefMap = .empty;
            if (index == 0) {
                var e_it = entry_map.iterator();
                while (e_it.next()) |e| {
                    try next_incoming.put(allocator, e.key_ptr.*, try defSetClone(allocator, e.value_ptr.*));
                }
            }
            var p_it = predecessors[index].iterator();
            while (p_it.next()) |p_entry| {
                const pred = p_entry.key_ptr.*;
                var out_it = outgoing[pred].iterator();
                while (out_it.next()) |o_entry| {
                    const gop = try next_incoming.getOrPut(allocator, o_entry.key_ptr.*);
                    if (!gop.found_existing) gop.value_ptr.* = .empty;
                    var d_it = o_entry.value_ptr.iterator();
                    while (d_it.next()) |d_entry| {
                        try gop.value_ptr.put(allocator, d_entry.key_ptr.*, {});
                    }
                }
            }

            var next_outgoing: RegDefMap = .empty;
            var in_it = next_incoming.iterator();
            while (in_it.next()) |in_entry| {
                try next_outgoing.put(allocator, in_entry.key_ptr.*, try defSetClone(allocator, in_entry.value_ptr.*));
            }
            var def_it = effect.definitions.iterator();
            while (def_it.next()) |d_entry| {
                const reg = d_entry.key_ptr.*;
                var single: DefinitionSet = .empty;
                try single.put(allocator, .{ .instruction = @intCast(index), .register = reg }, {});
                try next_outgoing.put(allocator, reg, single);
            }

            if (!regDefMapEqual(next_incoming, incoming[index]) or !regDefMapEqual(next_outgoing, outgoing[index])) {
                incoming[index] = next_incoming;
                outgoing[index] = next_outgoing;
                changed = true;
            }
        }
    }

    return .{ .incoming = incoming, .outgoing = outgoing };
}

fn computeDominators(
    allocator: Allocator,
    predecessors: []const IndexSet,
    reachable: IndexSet,
) ![]IndexSet {
    const dominators = try allocator.alloc(IndexSet, predecessors.len);
    for (0..predecessors.len) |idx| {
        dominators[idx] = .empty;
        if (!reachable.contains(idx)) {
            try dominators[idx].put(allocator, idx, {});
        } else {
            var r_it = reachable.iterator();
            while (r_it.next()) |r| try dominators[idx].put(allocator, r.key_ptr.*, {});
        }
    }
    if (predecessors.len == 0) return dominators;
    dominators[0] = .empty;
    try dominators[0].put(allocator, 0, {});

    var changed = true;
    while (changed) {
        changed = false;
        for (1..predecessors.len) |index| {
            if (!reachable.contains(index)) continue;
            var active_preds: std.ArrayList(usize) = .empty;
            var p_it = predecessors[index].iterator();
            while (p_it.next()) |p| {
                if (reachable.contains(p.key_ptr.*)) {
                    try active_preds.append(allocator, p.key_ptr.*);
                }
            }
            var next_val: IndexSet = .empty;
            if (active_preds.items.len > 0) {
                var first_it = dominators[active_preds.items[0]].iterator();
                while (first_it.next()) |d| {
                    const candidate = d.key_ptr.*;
                    var in_all = true;
                    for (active_preds.items[1..]) |pred| {
                        if (!dominators[pred].contains(candidate)) {
                            in_all = false;
                            break;
                        }
                    }
                    if (in_all) try next_val.put(allocator, candidate, {});
                }
            }
            try next_val.put(allocator, index, {});
            if (!indexSetEqual(next_val, dominators[index])) {
                dominators[index] = next_val;
                changed = true;
            }
        }
    }
    return dominators;
}

fn trapOnlyPath(
    allocator: Allocator,
    start: usize,
    effects: []const Effect,
    successors: []const IndexSet,
    visiting: IndexSet,
    memo: *std.AutoArrayHashMapUnmanaged(usize, bool),
) !bool {
    if (memo.get(start)) |cached| return cached;
    if (visiting.contains(start)) return false;
    const effect = effects[start];
    if (effect.trap) {
        try memo.put(allocator, start, true);
        return true;
    }
    const bk = effect.branch_kind;
    const disqualifying_branch = bk != null and (std.mem.eql(u8, bk.?, "call") or
        std.mem.eql(u8, bk.?, "conditional") or
        std.mem.eql(u8, bk.?, "indirect") or
        std.mem.eql(u8, bk.?, "return"));
    if (!effect.recognized or
        effect.memory or
        effect.uses.items.len > 0 or
        effect.definitions.count() > 0 or
        disqualifying_branch)
    {
        try memo.put(allocator, start, false);
        return false;
    }
    const targets = successors[start];
    if (targets.count() == 0) {
        try memo.put(allocator, start, false);
        return false;
    }
    var next_visiting: IndexSet = .empty;
    var v_it = visiting.iterator();
    while (v_it.next()) |v| try next_visiting.put(allocator, v.key_ptr.*, {});
    try next_visiting.put(allocator, start, {});

    var result = true;
    var t_it = targets.iterator();
    while (t_it.next()) |t| {
        if (!(try trapOnlyPath(allocator, t.key_ptr.*, effects, successors, next_visiting, memo))) {
            result = false;
            break;
        }
    }
    try memo.put(allocator, start, result);
    return result;
}

fn useSources(incoming: []const RegDefMap, instruction: usize, use: Use) DefinitionSet {
    return incoming[instruction].get(use.register) orelse .empty;
}

const GuardEvidence = struct {
    branch_instruction_index: usize,
    trap_successor_index: usize,
    normal_successor_index: usize,
    guarded_memory_instruction_index: usize,
    guarded_address_register: []const u8,
    compared_definition_register: []const u8,
};

fn structuralAddressGuardBranches(
    allocator: Allocator,
    effects: []const Effect,
    successors: []const IndexSet,
    incoming: []const RegDefMap,
    dominators: []const IndexSet,
    reachable: IndexSet,
) !struct {
    guard_branches: IndexSet,
    guard_definitions: DefinitionSet,
    evidence: []GuardEvidence,
} {
    var guard_branches: IndexSet = .empty;
    var guard_definitions: DefinitionSet = .empty;
    var evidence: std.ArrayList(GuardEvidence) = .empty;
    var trap_memo: std.AutoArrayHashMapUnmanaged(usize, bool) = .empty;

    for (effects, 0..) |effect, branch_index| {
        if (!reachable.contains(branch_index)) continue;
        if (effect.branch_kind == null or !std.mem.eql(u8, effect.branch_kind.?, "conditional")) continue;
        if (successors[branch_index].count() != 2) continue;

        var trap_succs: std.ArrayList(usize) = .empty;
        var s_it = successors[branch_index].iterator();
        while (s_it.next()) |s_entry| {
            const succ = s_entry.key_ptr.*;
            if (try trapOnlyPath(allocator, succ, effects, successors, .empty, &trap_memo)) {
                try trap_succs.append(allocator, succ);
            }
        }
        if (trap_succs.items.len != 1) continue;
        const trap_successor = trap_succs.items[0];
        var normal_successor: usize = 0;
        var s2_it = successors[branch_index].iterator();
        while (s2_it.next()) |s_entry| {
            if (s_entry.key_ptr.* != trap_successor) {
                normal_successor = s_entry.key_ptr.*;
                break;
            }
        }

        var control_sources: DefinitionSet = .empty;
        var valid_control = true;
        for (effect.uses.items) |use| {
            if (!std.mem.eql(u8, use.role, "control")) continue;
            const defs = useSources(incoming, branch_index, use);
            if (defs.count() != 1) {
                valid_control = false;
                break;
            }
            var d_it = defs.iterator();
            while (d_it.next()) |d| try control_sources.put(allocator, d.key_ptr.*, {});
        }
        if (!valid_control or control_sources.count() == 0) continue;

        var compared_sources: DefinitionSet = .empty;
        var cs_it = control_sources.iterator();
        while (cs_it.next()) |cs_entry| {
            const definition = cs_entry.key_ptr.*;
            if (definition.instruction < 0) {
                try compared_sources.put(allocator, definition, {});
                continue;
            }
            const producer = effects[@intCast(definition.instruction)];
            if (!std.mem.eql(u8, definition.register, "nzcv")) {
                try compared_sources.put(allocator, definition, {});
                continue;
            }
            var has_compare = false;
            for (producer.uses.items) |p_use| {
                if (!std.mem.eql(u8, p_use.role, "compare")) continue;
                has_compare = true;
                const sources = useSources(incoming, @intCast(definition.instruction), p_use);
                if (sources.count() != 1) {
                    valid_control = false;
                    break;
                }
                var src_it = sources.iterator();
                while (src_it.next()) |s| try compared_sources.put(allocator, s.key_ptr.*, {});
            }
            if (!has_compare or !valid_control) {
                valid_control = false;
                break;
            }
        }
        if (!valid_control or compared_sources.count() == 0) continue;

        var guarded_access: ?struct { mem_idx: usize, addr_reg: []const u8, cmp_reg: []const u8 } = null;
        for (effects, 0..) |mem_effect, memory_index| {
            if (!reachable.contains(memory_index) or
                !mem_effect.memory or
                !dominators[memory_index].contains(normal_successor) or
                !dominators[memory_index].contains(branch_index))
            {
                continue;
            }
            for (mem_effect.uses.items) |address_use| {
                if (!std.mem.eql(u8, address_use.role, "address")) continue;
                const sources = useSources(incoming, memory_index, address_use);
                if (sources.count() == 1) {
                    const only_src = sources.keys()[0];
                    if (compared_sources.contains(only_src)) {
                        guarded_access = .{
                            .mem_idx = memory_index,
                            .addr_reg = address_use.register,
                            .cmp_reg = only_src.register,
                        };
                        break;
                    }
                }
            }
            if (guarded_access != null) break;
        }
        const ga = guarded_access orelse continue;

        try guard_branches.put(allocator, branch_index, {});
        var cd_it = control_sources.iterator();
        while (cd_it.next()) |cd| try guard_definitions.put(allocator, cd.key_ptr.*, {});
        try evidence.append(allocator, .{
            .branch_instruction_index = branch_index,
            .trap_successor_index = trap_successor,
            .normal_successor_index = normal_successor,
            .guarded_memory_instruction_index = ga.mem_idx,
            .guarded_address_register = ga.addr_reg,
            .compared_definition_register = ga.cmp_reg,
        });
    }

    return .{
        .guard_branches = guard_branches,
        .guard_definitions = guard_definitions,
        .evidence = try evidence.toOwnedSlice(allocator),
    };
}

const ConsumersMap = std.ArrayHashMapUnmanaged(Definition, std.ArrayList(Consumer), DefinitionContext, true);

fn buildConsumers(
    allocator: Allocator,
    effects: []const Effect,
    incoming: []const RegDefMap,
    reachable: IndexSet,
) !ConsumersMap {
    var consumers: ConsumersMap = .empty;
    for (effects, 0..) |effect, index| {
        if (!reachable.contains(index)) continue;
        for (effect.uses.items) |use| {
            const definitions = useSources(incoming, index, use);
            const ambiguous = definitions.count() != 1;
            var d_it = definitions.iterator();
            while (d_it.next()) |d_entry| {
                const gop = try consumers.getOrPut(allocator, d_entry.key_ptr.*);
                if (!gop.found_existing) gop.value_ptr.* = .empty;
                try gop.value_ptr.append(allocator, .{
                    .instruction = index,
                    .use = use,
                    .ambiguous = ambiguous,
                });
            }
        }
    }
    return consumers;
}

const TraceMemo = std.ArrayHashMapUnmanaged(Definition, Trace, DefinitionContext, true);

fn traceDefinition(
    allocator: Allocator,
    definition: Definition,
    instructions: []const Instruction,
    effects: []const Effect,
    consumers: ConsumersMap,
    structural_guard_branches: IndexSet,
    structural_guard_definitions: DefinitionSet,
    stack: DefinitionSet,
    memo: *TraceMemo,
) !Trace {
    if (memo.get(definition)) |cached| {
        return cached.clone(allocator);
    }
    if (stack.contains(definition)) {
        var t: Trace = .{};
        try t.unknown_reasons.put(allocator, "cyclic producer-consumer path", {});
        return t;
    }
    const uses_list_opt = consumers.get(definition);
    if (uses_list_opt == null or uses_list_opt.?.items.len == 0) {
        var t: Trace = .{};
        try t.unknown_reasons.put(allocator, "definition has no proven consumer", {});
        return t;
    }
    const uses = uses_list_opt.?.items;
    var next_stack = try defSetClone(allocator, stack);
    try next_stack.put(allocator, definition, {});

    var result: Trace = .{};
    for (uses) |consumer| {
        const inst = instructions[consumer.instruction];
        if (consumer.ambiguous) {
            try result.unknown_reasons.put(
                allocator,
                "consumer has multiple reaching definitions at a CFG join",
                {},
            );
            try result.evidence.append(allocator, .{
                .offset = inst.offset,
                .instruction = inst.text,
                .role = "ambiguous_join",
            });
            continue;
        }
        const role = consumer.use.role;
        if (std.mem.eql(u8, role, "address")) {
            try result.roles.put(allocator, "address", {});
        } else if (std.mem.eql(u8, role, "escape")) {
            try result.unknown_reasons.put(
                allocator,
                "value escapes through untyped memory or a signatureless return",
                {},
            );
        } else if (std.mem.eql(u8, role, "control")) {
            if (structural_guard_branches.contains(consumer.instruction)) {
                try result.roles.put(allocator, "structural_guard", {});
            } else {
                try result.roles.put(allocator, "control", {});
            }
        } else if (std.mem.eql(u8, role, "compare")) {
            if (structural_guard_definitions.contains(.{
                .instruction = @intCast(consumer.instruction),
                .register = "nzcv",
            })) {
                try result.roles.put(allocator, "structural_guard", {});
            } else {
                try result.roles.put(allocator, "control", {});
            }
        } else if (std.mem.eql(u8, role, "unknown")) {
            try result.unknown_reasons.put(
                allocator,
                "value reaches an opaque instruction or call boundary",
                {},
            );
        } else if (!std.mem.eql(u8, role, "value")) {
            try result.unknown_reasons.put(
                allocator,
                try std.fmt.allocPrint(allocator, "unrecognized consumer role {s}", .{role}),
                {},
            );
        }

        try result.evidence.append(allocator, .{
            .offset = inst.offset,
            .instruction = inst.text,
            .role = role,
        });

        var propagation_targets: std.ArrayList([]const u8) = .empty;
        for (consumer.use.propagate_to) |target_register| {
            if (effects[consumer.instruction].definitions.contains(target_register)) {
                try propagation_targets.append(allocator, target_register);
            }
        }
        if (propagation_targets.items.len != consumer.use.propagate_to.len) {
            try result.unknown_reasons.put(
                allocator,
                "producer-consumer propagation target is not defined",
                {},
            );
        }
        var active_targets: std.ArrayList([]const u8) = .empty;
        for (propagation_targets.items) |target_register| {
            const td: Definition = .{
                .instruction = @intCast(consumer.instruction),
                .register = target_register,
            };
            if (consumers.get(td)) |c_list| {
                if (c_list.items.len > 0) {
                    try active_targets.append(allocator, target_register);
                }
            }
        }
        if (propagation_targets.items.len > 0 and active_targets.items.len == 0) {
            try result.unknown_reasons.put(
                allocator,
                "value transformer has no proven live output",
                {},
            );
        }
        for (active_targets.items) |target_register| {
            const target: Definition = .{
                .instruction = @intCast(consumer.instruction),
                .register = target_register,
            };
            const child = try traceDefinition(
                allocator,
                target,
                instructions,
                effects,
                consumers,
                structural_guard_branches,
                structural_guard_definitions,
                next_stack,
                memo,
            );
            try result.merge(allocator, child);
        }
    }

    try memo.put(allocator, definition, try result.clone(allocator));
    return result;
}

fn classifyTraceCategory(allocator: Allocator, trace: Trace) !struct { category: []const u8, reason: []const u8 } {
    if (trace.unknown_reasons.count() > 0) {
        var reasons: std.ArrayList([]const u8) = .empty;
        var it = trace.unknown_reasons.iterator();
        while (it.next()) |e| try reasons.append(allocator, e.key_ptr.*);
        std.mem.sort([]const u8, reasons.items, {}, struct {
            fn lessThan(_: void, a: []const u8, b: []const u8) bool {
                return std.mem.order(u8, a, b) == .lt;
            }
        }.lessThan);
        return .{
            .category = "unknown",
            .reason = try std.mem.join(allocator, "; ", reasons.items),
        };
    }

    var semantic_groups: std.ArrayList([]const u8) = .empty;
    if (trace.roles.contains("address")) try semantic_groups.append(allocator, "address");
    if (trace.roles.contains("structural_guard")) try semantic_groups.append(allocator, "structural_guard");
    if (trace.roles.contains("data")) try semantic_groups.append(allocator, "algorithmic");
    if (trace.roles.contains("control")) try semantic_groups.append(allocator, "control");

    if (semantic_groups.items.len == 0) {
        return .{ .category = "unknown", .reason = "no terminal semantic consumer was proven" };
    }
    if (semantic_groups.items.len == 1 and std.mem.eql(u8, semantic_groups.items[0], "control")) {
        return .{
            .category = "unknown",
            .reason = "control semantics are not proven bounds/check or algorithmic data",
        };
    }
    if (semantic_groups.items.len > 1) {
        std.mem.sort([]const u8, semantic_groups.items, {}, struct {
            fn lessThan(_: void, a: []const u8, b: []const u8) bool {
                return std.mem.order(u8, a, b) == .lt;
            }
        }.lessThan);
        const joined = try std.mem.join(allocator, ", ", semantic_groups.items);
        return .{
            .category = "mixed",
            .reason = try std.fmt.allocPrint(
                allocator,
                "producer has proven consumers in multiple semantic categories: {s}",
                .{joined},
            ),
        };
    }
    const only = semantic_groups.items[0];
    if (std.mem.eql(u8, only, "address")) {
        return .{
            .category = "address_generation",
            .reason = "all proven producer-consumer paths terminate in memory-address operands",
        };
    }
    if (std.mem.eql(u8, only, "structural_guard")) {
        return .{
            .category = "structural_address_guard",
            .reason = "flags/control structurally guard a dominated eventual address " ++
                "and the other complete CFG path reaches an architectural trap; " ++
                "linear-memory limit or removable-check semantics are not proven",
        };
    }
    return .{
        .category = "algorithmic_alu",
        .reason = "all terminal uses have explicit typed non-address data provenance",
    };
}

pub const SampleEntry = struct {
    offset: i64,
    samples: i64,
};

pub const AnalyzeOptions = struct {
    broad_classes: []const []const u8,
    samples_by_offset: []const SampleEntry = &.{},
    total_run_samples: i64,
    address_base: i64 = 0,
    global_attributed_samples: ?i64 = null,
};

const CachedDecision = struct {
    category: []const u8,
    reason: []const u8,
    trace: Trace,
};

pub fn analyzeInstructionStream(
    allocator: Allocator,
    instructions: []const InstructionInput,
    options: AnalyzeOptions,
    diag: *Diagnostic,
) ProvenanceError!Value {
    if (options.total_run_samples < 0) {
        return diag.fail(allocator, "total_run_samples must be a nonnegative integer", .{});
    }
    if (options.total_run_samples == 0) {
        return diag.fail(allocator, "total_run_samples must be positive", .{});
    }
    const global_attributed = options.global_attributed_samples orelse options.total_run_samples;
    if (global_attributed < 0) {
        return diag.fail(allocator, "global_attributed_samples must be a nonnegative integer", .{});
    }
    if (global_attributed > options.total_run_samples) {
        return diag.fail(allocator, "global_attributed_samples must be within total_run_samples", .{});
    }

    const normalized = try normalizeInstructions(allocator, instructions, options.address_base, diag);
    if (options.broad_classes.len != normalized.len) {
        return diag.fail(allocator, "broad instruction classes must align exactly with disassembly", .{});
    }

    var known_offsets: std.AutoArrayHashMapUnmanaged(i64, void) = .empty;
    for (normalized) |inst| try known_offsets.put(allocator, inst.offset, {});

    var samples_map: std.AutoArrayHashMapUnmanaged(i64, i64) = .empty;
    var invalid_offsets: std.ArrayList(i64) = .empty;
    var mapped_instruction_samples: i64 = 0;
    for (options.samples_by_offset) |entry| {
        if (entry.samples < 0) {
            return diag.fail(allocator, "sample count at offset {d} must be a nonnegative integer", .{entry.offset});
        }
        mapped_instruction_samples += entry.samples;
        if (!known_offsets.contains(entry.offset)) {
            try invalid_offsets.append(allocator, entry.offset);
        } else {
            try samples_map.put(allocator, entry.offset, entry.samples);
        }
    }
    if (invalid_offsets.items.len > 0) {
        return diag.fail(
            allocator,
            "sample mappings must use exact nonnegative instruction offsets; invalid offsets: {d}",
            .{invalid_offsets.items[0]},
        );
    }
    if (mapped_instruction_samples > global_attributed) {
        return diag.fail(allocator, "mapped instruction samples exceed globally attributed samples", .{});
    }

    const effects = try allocator.alloc(Effect, normalized.len);
    for (normalized, 0..) |inst, idx| {
        effects[idx] = try instructionEffect(allocator, inst);
    }
    const cfg = try buildCfg(allocator, normalized, effects);
    const reachable = try computeReachable(allocator, cfg.successors);
    const reaching = try reachingDefinitions(allocator, effects, cfg.predecessors, reachable);
    const dominators = try computeDominators(allocator, cfg.predecessors, reachable);
    const guards = try structuralAddressGuardBranches(
        allocator,
        effects,
        cfg.successors,
        reaching.incoming,
        dominators,
        reachable,
    );
    const consumers = try buildConsumers(allocator, effects, reaching.incoming, reachable);

    var memo: TraceMemo = .empty;
    var decision_cache: std.AutoArrayHashMapUnmanaged(usize, CachedDecision) = .empty;

    const Helper = struct {
        alloc: Allocator,
        norm: []const Instruction,
        effs: []const Effect,
        reach: IndexSet,
        cons: ConsumersMap,
        guard_branches: IndexSet,
        guard_defs: DefinitionSet,
        trace_memo: *TraceMemo,
        dec_cache: *std.AutoArrayHashMapUnmanaged(usize, CachedDecision),
        s_map: std.AutoArrayHashMapUnmanaged(i64, i64),
        total_samples: i64,

        fn classifyIndex(self: *@This(), index: usize) !CachedDecision {
            if (self.dec_cache.get(index)) |cached| return cached;
            const effect = self.effs[index];
            var decision: CachedDecision = undefined;
            if (!self.reach.contains(index)) {
                const reason = "instruction is unreachable in the recovered CFG";
                var tr: Trace = .{};
                try tr.unknown_reasons.put(self.alloc, reason, {});
                decision = .{ .category = "unknown", .reason = reason, .trace = tr };
            } else if (!effect.recognized) {
                const reason = "instruction semantics are not recognized conservatively";
                var tr: Trace = .{};
                try tr.unknown_reasons.put(self.alloc, reason, {});
                decision = .{ .category = "unknown", .reason = reason, .trace = tr };
            } else if (effect.memory) {
                const reason = "combined memory and value-update costs cannot be isolated as pure ALU";
                var tr: Trace = .{};
                try tr.unknown_reasons.put(self.alloc, reason, {});
                decision = .{ .category = "unknown", .reason = reason, .trace = tr };
            } else {
                var definitions: std.StringArrayHashMapUnmanaged(void) = .empty;
                for ([_]std.StringArrayHashMapUnmanaged(void){ effect.primary_definitions, effect.secondary_definitions }) |set| {
                    var it = set.iterator();
                    while (it.next()) |e| {
                        const reg = e.key_ptr.*;
                        if (self.cons.get(.{ .instruction = @intCast(index), .register = reg })) |c_list| {
                            if (c_list.items.len > 0) try definitions.put(self.alloc, reg, {});
                        }
                    }
                }
                if (definitions.count() == 0) {
                    var p_it = effect.primary_definitions.iterator();
                    while (p_it.next()) |e| try definitions.put(self.alloc, e.key_ptr.*, {});
                }
                var combined: Trace = .{};
                var d_it = definitions.iterator();
                while (d_it.next()) |e| {
                    const sub = try traceDefinition(
                        self.alloc,
                        .{ .instruction = @intCast(index), .register = e.key_ptr.* },
                        self.norm,
                        self.effs,
                        self.cons,
                        self.guard_branches,
                        self.guard_defs,
                        .empty,
                        self.trace_memo,
                    );
                    try combined.merge(self.alloc, sub);
                }
                const cat_res = try classifyTraceCategory(self.alloc, combined);
                decision = .{ .category = cat_res.category, .reason = cat_res.reason, .trace = combined };
            }
            try self.dec_cache.put(self.alloc, index, decision);
            return decision;
        }

        fn aggregate(self: *@This(), indices: []const usize, share_name: []const u8) !ObjectMap {
            var static_counts: std.StringArrayHashMapUnmanaged(i64) = .empty;
            var sample_counts: std.StringArrayHashMapUnmanaged(i64) = .empty;
            for (CATEGORIES) |cat| {
                try static_counts.put(self.alloc, cat, 0);
                try sample_counts.put(self.alloc, cat, 0);
            }

            const ReasonKey = struct {
                category: []const u8,
                reason: []const u8,
            };
            const ReasonKeyCtx = struct {
                pub fn hash(_: @This(), k: ReasonKey) u32 {
                    var h = std.hash.Wyhash.init(0);
                    h.update(k.category);
                    h.update(&.{0});
                    h.update(k.reason);
                    return @truncate(h.final());
                }
                pub fn eql(_: @This(), a: ReasonKey, b: ReasonKey, _: usize) bool {
                    return std.mem.eql(u8, a.category, b.category) and std.mem.eql(u8, a.reason, b.reason);
                }
            };
            var reason_counts: std.ArrayHashMapUnmanaged(ReasonKey, i64, ReasonKeyCtx, true) = .empty;

            const SampledRow = struct {
                offset: i64,
                samples: i64,
                val: Value,
            };
            var sampled_rows: std.ArrayList(SampledRow) = .empty;

            for (indices) |index| {
                const inst = self.norm[index];
                const dec = try self.classifyIndex(index);
                static_counts.getPtr(dec.category).?.* += 1;
                const samples = self.s_map.get(inst.offset) orelse 0;
                sample_counts.getPtr(dec.category).?.* += samples;
                const rgop = try reason_counts.getOrPut(self.alloc, .{ .category = dec.category, .reason = dec.reason });
                if (!rgop.found_existing) rgop.value_ptr.* = 0;
                rgop.value_ptr.* += 1;

                if (samples > 0) {
                    var ev_arr = Array.init(self.alloc);
                    const ev_limit = @min(dec.trace.evidence.items.len, 12);
                    for (dec.trace.evidence.items[0..ev_limit]) |ev| {
                        var ev_obj: ObjectMap = .empty;
                        try ev_obj.put(self.alloc, "instruction", .{ .string = ev.instruction });
                        try ev_obj.put(self.alloc, "offset", .{ .integer = ev.offset });
                        try ev_obj.put(self.alloc, "role", .{ .string = ev.role });
                        try ev_arr.append(.{ .object = ev_obj });
                    }
                    var row_obj: ObjectMap = .empty;
                    try row_obj.put(self.alloc, "address", .{ .integer = inst.address });
                    try row_obj.put(self.alloc, "category", .{ .string = dec.category });
                    try row_obj.put(self.alloc, "instruction", .{ .string = inst.text });
                    try row_obj.put(self.alloc, "offset", .{ .integer = inst.offset });
                    try row_obj.put(self.alloc, "path_evidence", .{ .array = ev_arr });
                    try row_obj.put(
                        self.alloc,
                        "percent_of_run",
                        .{ .float = 100.0 * @as(f64, @floatFromInt(samples)) / @as(f64, @floatFromInt(self.total_samples)) },
                    );
                    try row_obj.put(self.alloc, "reason", .{ .string = dec.reason });
                    try row_obj.put(self.alloc, "samples", .{ .integer = samples });
                    try sampled_rows.append(self.alloc, .{
                        .offset = inst.offset,
                        .samples = samples,
                        .val = .{ .object = row_obj },
                    });
                }
            }

            std.mem.sort(SampledRow, sampled_rows.items, {}, struct {
                fn lessThan(_: void, a: SampledRow, b: SampledRow) bool {
                    if (a.samples != b.samples) return a.samples > b.samples;
                    return a.offset < b.offset;
                }
            }.lessThan);
            var sampled_arr = Array.init(self.alloc);
            for (sampled_rows.items) |r| try sampled_arr.append(r.val);

            var universe_samples: i64 = 0;
            for (CATEGORIES) |cat| universe_samples += sample_counts.get(cat).?;
            var proven_samples: i64 = 0;
            for (PROVEN_CATEGORIES) |cat| proven_samples += sample_counts.get(cat).?;

            var categories_obj: ObjectMap = .empty;
            for (CATEGORIES) |cat| {
                const sc = sample_counts.get(cat).?;
                const stc = static_counts.get(cat).?;
                var c_obj: ObjectMap = .empty;
                try c_obj.put(self.alloc, "samples", .{ .integer = sc });
                try c_obj.put(
                    self.alloc,
                    "percent_of_run",
                    .{ .float = 100.0 * @as(f64, @floatFromInt(sc)) / @as(f64, @floatFromInt(self.total_samples)) },
                );
                try c_obj.put(
                    self.alloc,
                    share_name,
                    .{ .float = if (universe_samples > 0)
                        100.0 * @as(f64, @floatFromInt(sc)) / @as(f64, @floatFromInt(universe_samples))
                    else
                        0.0 },
                );
                try c_obj.put(self.alloc, "static_instructions", .{ .integer = stc });
                try categories_obj.put(self.alloc, cat, .{ .object = c_obj });
            }

            const unknown_samples = sample_counts.get("unknown").?;
            const mixed_samples = sample_counts.get("mixed").?;
            var coverage_obj: ObjectMap = .empty;
            try coverage_obj.put(self.alloc, "proven_samples", .{ .integer = proven_samples });
            try coverage_obj.put(
                self.alloc,
                "proven_percent",
                .{ .float = if (universe_samples > 0)
                    100.0 * @as(f64, @floatFromInt(proven_samples)) / @as(f64, @floatFromInt(universe_samples))
                else
                    0.0 },
            );
            try coverage_obj.put(self.alloc, "unknown_samples", .{ .integer = unknown_samples });
            try coverage_obj.put(
                self.alloc,
                "unknown_percent_of_run",
                .{ .float = 100.0 * @as(f64, @floatFromInt(unknown_samples)) / @as(f64, @floatFromInt(self.total_samples)) },
            );
            try coverage_obj.put(self.alloc, "mixed_samples", .{ .integer = mixed_samples });
            try coverage_obj.put(
                self.alloc,
                "mixed_percent_of_run",
                .{ .float = 100.0 * @as(f64, @floatFromInt(mixed_samples)) / @as(f64, @floatFromInt(self.total_samples)) },
            );

            const ReasonEntry = struct {
                category: []const u8,
                reason: []const u8,
                count: i64,
            };
            var r_list: std.ArrayList(ReasonEntry) = .empty;
            var r_it = reason_counts.iterator();
            while (r_it.next()) |entry| {
                try r_list.append(self.alloc, .{
                    .category = entry.key_ptr.category,
                    .reason = entry.key_ptr.reason,
                    .count = entry.value_ptr.*,
                });
            }
            std.mem.sort(ReasonEntry, r_list.items, {}, struct {
                fn lessThan(_: void, a: ReasonEntry, b: ReasonEntry) bool {
                    const c_ord = std.mem.order(u8, a.category, b.category);
                    if (c_ord != .eq) return c_ord == .lt;
                    return std.mem.order(u8, a.reason, b.reason) == .lt;
                }
            }.lessThan);
            var reason_arr = Array.init(self.alloc);
            for (r_list.items) |re| {
                var re_obj: ObjectMap = .empty;
                try re_obj.put(self.alloc, "category", .{ .string = re.category });
                try re_obj.put(self.alloc, "reason", .{ .string = re.reason });
                try re_obj.put(self.alloc, "static_instructions", .{ .integer = re.count });
                try reason_arr.append(.{ .object = re_obj });
            }

            var out: ObjectMap = .empty;
            try out.put(self.alloc, "samples", .{ .integer = universe_samples });
            try out.put(self.alloc, "partition_samples", .{ .integer = universe_samples });
            try out.put(self.alloc, "static_instructions", .{ .integer = @intCast(indices.len) });
            try out.put(self.alloc, "categories", .{ .object = categories_obj });
            try out.put(self.alloc, "coverage", .{ .object = coverage_obj });
            try out.put(self.alloc, "sampled_instructions", .{ .array = sampled_arr });
            try out.put(self.alloc, "reason_counts", .{ .array = reason_arr });
            return out;
        }
    };

    var helper: Helper = .{
        .alloc = allocator,
        .norm = normalized,
        .effs = effects,
        .reach = reachable,
        .cons = consumers,
        .guard_branches = guards.guard_branches,
        .guard_defs = guards.guard_definitions,
        .trace_memo = &memo,
        .dec_cache = &decision_cache,
        .s_map = samples_map,
        .total_samples = options.total_run_samples,
    };

    var legacy_indices: std.ArrayList(usize) = .empty;
    var common_indices: std.ArrayList(usize) = .empty;
    for (options.broad_classes, 0..) |broad_class, idx| {
        const is_alu = std.mem.eql(u8, broad_class, "alu");
        if (is_alu) try legacy_indices.append(allocator, idx);
        if (is_alu or effects[idx].gating_candidate or !effects[idx].recognized) {
            try common_indices.append(allocator, idx);
        }
    }

    const legacy = try helper.aggregate(legacy_indices.items, "percent_of_broad_alu");
    const common = try helper.aggregate(common_indices.items, "percent_of_common_universe");

    const common_samples = common.get("samples").?.integer;
    const legacy_samples = legacy.get("samples").?.integer;
    const excluded_instruction_samples = mapped_instruction_samples - common_samples;
    if (excluded_instruction_samples < 0) {
        return diag.fail(allocator, "common gating universe exceeds mapped instruction samples", .{});
    }

    var legacy_cov = legacy.get("coverage").?.object;
    try legacy_cov.put(allocator, "proven_percent_of_broad_alu", legacy_cov.get("proven_percent").?);

    var acct_obj: ObjectMap = .empty;
    try acct_obj.put(allocator, "mapped_instruction_samples", .{ .integer = mapped_instruction_samples });
    try acct_obj.put(allocator, "common_candidate_samples", .{ .integer = common_samples });
    try acct_obj.put(
        allocator,
        "conclusively_excluded_non_candidate_samples",
        .{ .integer = excluded_instruction_samples },
    );

    var global_obj: ObjectMap = .empty;
    try global_obj.put(allocator, "attributed_samples", .{ .integer = global_attributed });
    try global_obj.put(allocator, "unattributed_samples", .{ .integer = options.total_run_samples - global_attributed });
    try global_obj.put(allocator, "total_samples", .{ .integer = options.total_run_samples });

    var ext_arr = Array.init(allocator);
    for (cfg.external_targets) |et| {
        var et_obj: ObjectMap = .empty;
        try et_obj.put(allocator, "offset", .{ .integer = et.offset });
        try et_obj.put(allocator, "instruction", .{ .string = et.instruction });
        try et_obj.put(allocator, "target", .{ .integer = et.target });
        try ext_arr.append(.{ .object = et_obj });
    }

    var guard_ev_arr = Array.init(allocator);
    for (guards.evidence) |item| {
        var ge_obj: ObjectMap = .empty;
        try ge_obj.put(allocator, "branch_instruction_index", .{ .integer = @intCast(item.branch_instruction_index) });
        try ge_obj.put(allocator, "trap_successor_index", .{ .integer = @intCast(item.trap_successor_index) });
        try ge_obj.put(allocator, "normal_successor_index", .{ .integer = @intCast(item.normal_successor_index) });
        try ge_obj.put(allocator, "guarded_memory_instruction_index", .{ .integer = @intCast(item.guarded_memory_instruction_index) });
        try ge_obj.put(allocator, "guarded_address_register", .{ .string = item.guarded_address_register });
        try ge_obj.put(allocator, "compared_definition_register", .{ .string = item.compared_definition_register });
        try ge_obj.put(allocator, "branch_offset", .{ .integer = normalized[item.branch_instruction_index].offset });
        try ge_obj.put(allocator, "branch_instruction", .{ .string = normalized[item.branch_instruction_index].text });
        try ge_obj.put(allocator, "guarded_memory_offset", .{ .integer = normalized[item.guarded_memory_instruction_index].offset });
        try ge_obj.put(allocator, "guarded_memory_instruction", .{ .string = normalized[item.guarded_memory_instruction_index].text });
        try guard_ev_arr.append(.{ .object = ge_obj });
    }

    var cfg_obj: ObjectMap = .empty;
    try cfg_obj.put(allocator, "reachable_instructions", .{ .integer = @intCast(reachable.count()) });
    try cfg_obj.put(allocator, "external_branch_targets", .{ .array = ext_arr });
    try cfg_obj.put(allocator, "structural_address_guard_branches", .{ .integer = @intCast(guards.guard_branches.count()) });
    try cfg_obj.put(allocator, "structural_address_guard_evidence", .{ .array = guard_ev_arr });

    var soundness_obj: ObjectMap = .empty;
    try soundness_obj.put(
        allocator,
        "register_aliases",
        .{ .string = "Wn and Xn share one architectural value; W writes replace the full tracked value because AArch64 zero-extends them." },
    );
    try soundness_obj.put(
        allocator,
        "joins_and_loops",
        .{ .string = "A use with multiple reaching definitions, or a cyclic path without a proven terminal role, remains unknown." },
    );
    try soundness_obj.put(
        allocator,
        "calls",
        .{ .string = "AAPCS64 possible x0-x7 arguments are opaque uses; x0-x18, x30, and NZCV are clobbered. No engine-specific register meaning is assumed." },
    );
    try soundness_obj.put(
        allocator,
        "structural_address_guard",
        .{ .string = "A structural guard requires a complete CFG trap path plus a dominated eventual address sharing the compared definition. It does not prove a linear-memory limit, engine bounds check, or removable check." },
    );
    try soundness_obj.put(
        allocator,
        "mixed",
        .{ .string = "Only independently proven semantic roles can produce mixed. Untyped stores, signatureless returns, and unproven control remain unknown escapes instead." },
    );

    var out: ObjectMap = .empty;
    try out.put(allocator, "schema_version", .{ .integer = SCHEMA_VERSION });
    try out.put(allocator, "kind", .{ .string = ANALYSIS_KIND });
    try out.put(allocator, "status", .{ .string = "measured" });
    try out.put(allocator, "broad_class", .{ .string = "all_alu" });
    try out.put(allocator, "total_run_samples", .{ .integer = options.total_run_samples });
    try out.put(allocator, "broad_alu_samples", .{ .integer = legacy_samples });
    try out.put(
        allocator,
        "broad_alu_percent_of_run",
        .{ .float = 100.0 * @as(f64, @floatFromInt(legacy_samples)) / @as(f64, @floatFromInt(options.total_run_samples)) },
    );
    try out.put(allocator, "partition_samples", legacy.get("partition_samples").?);
    try out.put(allocator, "instruction_count", .{ .integer = @intCast(normalized.len) });
    try out.put(allocator, "broad_alu_static_instructions", legacy.get("static_instructions").?);
    try out.put(allocator, "categories", legacy.get("categories").?);
    try out.put(allocator, "coverage", .{ .object = legacy_cov });
    try out.put(allocator, "common_gating_universe", .{ .object = common });
    try out.put(allocator, "instruction_sample_accounting", .{ .object = acct_obj });
    try out.put(allocator, "global_sample_mapping", .{ .object = global_obj });
    try out.put(allocator, "cfg", .{ .object = cfg_obj });
    try out.put(allocator, "sampled_instructions", legacy.get("sampled_instructions").?);
    try out.put(allocator, "reason_counts", legacy.get("reason_counts").?);
    try out.put(allocator, "soundness", .{ .object = soundness_obj });
    return .{ .object = out };
}

pub fn unavailableAnalysis(
    allocator: Allocator,
    reason: []const u8,
    total_run_samples: ?i64,
) !Value {
    var out: ObjectMap = .empty;
    try out.put(allocator, "schema_version", .{ .integer = SCHEMA_VERSION });
    try out.put(allocator, "kind", .{ .string = ANALYSIS_KIND });
    try out.put(allocator, "status", .{ .string = "unavailable" });
    try out.put(allocator, "reason", .{ .string = reason });
    try out.put(
        allocator,
        "total_run_samples",
        if (total_run_samples) |t| .{ .integer = t } else .null,
    );
    return .{ .object = out };
}

fn requireMapping(
    allocator: Allocator,
    val: ?Value,
    name: []const u8,
    label: []const u8,
    diag: *Diagnostic,
) ProvenanceError!ObjectMap {
    const v = val orelse return diag.fail(allocator, "{s} {s} is missing or invalid", .{ name, label });
    if (v != .object) return diag.fail(allocator, "{s} {s} is missing or invalid", .{ name, label });
    return v.object;
}

fn requireNonnegInt(
    allocator: Allocator,
    val: ?Value,
    label: []const u8,
    diag: *Diagnostic,
) ProvenanceError!i64 {
    const v = val orelse return diag.fail(allocator, "{s} must be a nonnegative integer", .{label});
    if (v != .integer or v.integer < 0) {
        return diag.fail(allocator, "{s} must be a nonnegative integer", .{label});
    }
    return v.integer;
}

fn jsonFiniteFloat(val: ?Value) ?f64 {
    const v = val orelse return null;
    const f: f64 = switch (v) {
        .float => |fl| fl,
        .integer => |i| @floatFromInt(i),
        else => return null,
    };
    if (!std.math.isFinite(f)) return null;
    return f;
}

fn isClose(a: f64, b: f64, rel_tol: f64, abs_tol: f64) bool {
    const diff = @abs(a - b);
    const bound = @max(rel_tol * @max(@abs(a), @abs(b)), abs_tol);
    return diff <= bound;
}

pub fn validateAnalysisSamples(
    allocator: Allocator,
    analysis_val: Value,
    name: []const u8,
    diag: *Diagnostic,
) ProvenanceError!void {
    if (analysis_val != .object) {
        return diag.fail(allocator, "{s} has an invalid analysis identity", .{name});
    }
    const analysis = analysis_val.object;
    const sv = analysis.get("schema_version") orelse return diag.fail(allocator, "{s} has an invalid analysis identity", .{name});
    const kind = analysis.get("kind") orelse return diag.fail(allocator, "{s} has an invalid analysis identity", .{name});
    const status = analysis.get("status") orelse return diag.fail(allocator, "{s} has an invalid analysis identity", .{name});
    if (sv != .integer or sv.integer != SCHEMA_VERSION or
        kind != .string or !std.mem.eql(u8, kind.string, ANALYSIS_KIND) or
        status != .string or !std.mem.eql(u8, status.string, "measured"))
    {
        return diag.fail(allocator, "{s} has an invalid analysis identity", .{name});
    }

    const total = try requireNonnegInt(
        allocator,
        analysis.get("total_run_samples"),
        try std.fmt.allocPrint(allocator, "{s} total", .{name}),
        diag,
    );
    if (total == 0) {
        return diag.fail(allocator, "{s} total_run_samples must be positive", .{name});
    }

    const global_mapping = try requireMapping(allocator, analysis.get("global_sample_mapping"), name, "global mapping", diag);
    const attributed = try requireNonnegInt(
        allocator,
        global_mapping.get("attributed_samples"),
        try std.fmt.allocPrint(allocator, "{s} globally attributed", .{name}),
        diag,
    );
    const unattributed = try requireNonnegInt(
        allocator,
        global_mapping.get("unattributed_samples"),
        try std.fmt.allocPrint(allocator, "{s} globally unattributed", .{name}),
        diag,
    );
    const gm_total = global_mapping.get("total_samples") orelse Value.null;
    if (gm_total != .integer or gm_total.integer != total or attributed + unattributed != total) {
        return diag.fail(allocator, "{s} global sample mapping does not reconcile", .{name});
    }

    const common = try requireMapping(allocator, analysis.get("common_gating_universe"), name, "common universe", diag);
    const common_samples = try requireNonnegInt(
        allocator,
        common.get("samples"),
        try std.fmt.allocPrint(allocator, "{s} common samples", .{name}),
        diag,
    );
    const legacy_samples = try requireNonnegInt(
        allocator,
        analysis.get("broad_alu_samples"),
        try std.fmt.allocPrint(allocator, "{s} legacy samples", .{name}),
        diag,
    );

    for ([_]struct { label: []const u8, summary: ObjectMap, samples: i64 }{
        .{ .label = "legacy", .summary = analysis, .samples = legacy_samples },
        .{ .label = "common", .summary = common, .samples = common_samples },
    }) |item| {
        const ps = item.summary.get("partition_samples") orelse Value.null;
        if (ps != .integer or ps.integer != item.samples) {
            return diag.fail(allocator, "{s} {s} sample partitions do not reconcile", .{ name, item.label });
        }
        const categories = try requireMapping(
            allocator,
            item.summary.get("categories"),
            name,
            try std.fmt.allocPrint(allocator, "{s} categories", .{item.label}),
            diag,
        );
        if (categories.count() != CATEGORIES.len) {
            return diag.fail(allocator, "{s} {s} categories are incomplete", .{ name, item.label });
        }
        var partition: i64 = 0;
        for (CATEGORIES) |cat| {
            const values = try requireMapping(
                allocator,
                categories.get(cat),
                name,
                try std.fmt.allocPrint(allocator, "{s} {s}", .{ item.label, cat }),
                diag,
            );
            const count = try requireNonnegInt(
                allocator,
                values.get("samples"),
                try std.fmt.allocPrint(allocator, "{s} {s} {s} samples", .{ name, item.label, cat }),
                diag,
            );
            partition += count;
            const share = jsonFiniteFloat(values.get("percent_of_run")) orelse {
                return diag.fail(allocator, "{s} {s} {s} share disagrees with sample counts", .{ name, item.label, cat });
            };
            const expected_share = 100.0 * @as(f64, @floatFromInt(count)) / @as(f64, @floatFromInt(total));
            if (!isClose(share, expected_share, 1e-12, 1e-9)) {
                return diag.fail(allocator, "{s} {s} {s} share disagrees with sample counts", .{ name, item.label, cat });
            }
        }
        if (partition != item.samples) {
            return diag.fail(allocator, "{s} {s} category samples do not reconcile", .{ name, item.label });
        }
    }

    const accounting = try requireMapping(
        allocator,
        analysis.get("instruction_sample_accounting"),
        name,
        "instruction accounting",
        diag,
    );
    const mapped = try requireNonnegInt(
        allocator,
        accounting.get("mapped_instruction_samples"),
        try std.fmt.allocPrint(allocator, "{s} mapped instructions", .{name}),
        diag,
    );
    const excluded = try requireNonnegInt(
        allocator,
        accounting.get("conclusively_excluded_non_candidate_samples"),
        try std.fmt.allocPrint(allocator, "{s} excluded instructions", .{name}),
        diag,
    );
    const common_cand = accounting.get("common_candidate_samples") orelse Value.null;
    if (common_cand != .integer or
        common_cand.integer != common_samples or
        common_samples + excluded != mapped or
        legacy_samples > common_samples)
    {
        return diag.fail(allocator, "{s} instruction sample accounting does not reconcile", .{name});
    }
    if (mapped > attributed) {
        return diag.fail(allocator, "{s} mapped instruction samples exceed globally attributed samples", .{name});
    }

    if (analysis.get("sample_mapping")) |sm_val| {
        const function_mapping = try requireMapping(allocator, sm_val, name, "function mapping", diag);
        const function_mapped = if (function_mapping.get("mapped_function_samples")) |v|
            try requireNonnegInt(allocator, v, try std.fmt.allocPrint(allocator, "{s} mapped function samples", .{name}), diag)
        else
            mapped;
        const unresolved = if (function_mapping.get("unresolved_function_samples")) |v|
            try requireNonnegInt(allocator, v, try std.fmt.allocPrint(allocator, "{s} unresolved function samples", .{name}), diag)
        else
            0;
        if (function_mapped != mapped or mapped + unresolved > attributed) {
            return diag.fail(allocator, "{s} function sample mapping does not reconcile", .{name});
        }
    }
}

pub fn compareEngineAnalyses(
    allocator: Allocator,
    wamr_val: Value,
    wasmtime_val: Value,
    threshold_pct: f64,
    diag: *Diagnostic,
) ProvenanceError!Value {
    if (!std.math.isFinite(threshold_pct) or threshold_pct < 0) {
        return diag.fail(allocator, "gate threshold must be finite and nonnegative", .{});
    }

    var unavailable: std.ArrayList([]const u8) = .empty;
    for ([_]struct { name: []const u8, val: Value }{
        .{ .name = "wamr", .val = wamr_val },
        .{ .name = "wasmtime", .val = wasmtime_val },
    }) |entry| {
        const st = if (entry.val == .object) entry.val.object.get("status") else null;
        const is_measured = st != null and st.? == .string and std.mem.eql(u8, st.?.string, "measured");
        if (!is_measured) {
            const r_val = if (entry.val == .object) entry.val.object.get("reason") else null;
            const reason = if (r_val != null and r_val.? == .string) r_val.?.string else "analysis unavailable";
            try unavailable.append(allocator, try std.fmt.allocPrint(allocator, "{s}: {s}", .{ entry.name, reason }));
        }
    }
    if (unavailable.items.len > 0) {
        var out: ObjectMap = .empty;
        try out.put(allocator, "threshold_percentage_points", .{ .float = threshold_pct });
        try out.put(allocator, "status", .{ .string = "blocked" });
        try out.put(allocator, "optimization_authorized", .{ .bool = false });
        try out.put(allocator, "reason", .{ .string = try std.mem.join(allocator, "; ", unavailable.items) });
        try out.put(allocator, "categories", .{ .object = .empty });
        return .{ .object = out };
    }

    try validateAnalysisSamples(allocator, wamr_val, "wamr", diag);
    try validateAnalysisSamples(allocator, wasmtime_val, "wasmtime", diag);

    const wamr = wamr_val.object;
    const wasmtime = wasmtime_val.object;
    const wasmtime_common = wasmtime.get("common_gating_universe").?.object;
    const wasmtime_global_unattributed = wasmtime.get("global_sample_mapping").?.object.get("unattributed_samples").?.integer;
    const wasmtime_instruction_unresolved = if (wasmtime.get("sample_mapping")) |sm|
        if (sm == .object and sm.object.get("unresolved_function_samples") != null)
            sm.object.get("unresolved_function_samples").?.integer
        else
            0
    else
        0;

    const wt_cats = wasmtime_common.get("categories").?.object;
    const wt_unknown_samples = wt_cats.get("unknown").?.object.get("samples").?.integer;
    const wt_mixed_samples = wt_cats.get("mixed").?.object.get("samples").?.integer;
    const wt_total = wasmtime.get("total_run_samples").?.integer;
    const wamr_total = wamr.get("total_run_samples").?.integer;

    const wasmtime_unknown_upper = 100.0 *
        @as(f64, @floatFromInt(wt_unknown_samples + wt_mixed_samples + wasmtime_instruction_unresolved + wasmtime_global_unattributed)) /
        @as(f64, @floatFromInt(wt_total));

    var categories: ObjectMap = .empty;
    var cleared = Array.init(allocator);
    for (GATE_CATEGORIES) |category| {
        const wamr_category = wamr.get("common_gating_universe").?.object.get("categories").?.object.get(category).?.object;
        const wasmtime_category = wt_cats.get(category).?.object;
        const wamr_s = wamr_category.get("samples").?.integer;
        const wt_s = wasmtime_category.get("samples").?.integer;
        const wamr_share = 100.0 * @as(f64, @floatFromInt(wamr_s)) / @as(f64, @floatFromInt(wamr_total));
        const wasmtime_share = 100.0 * @as(f64, @floatFromInt(wt_s)) / @as(f64, @floatFromInt(wt_total));
        const conservative_wasmtime_upper = wasmtime_share + wasmtime_unknown_upper;
        const conservative_headroom = wamr_share - conservative_wasmtime_upper;
        const clears = conservative_headroom >= threshold_pct;
        if (clears) try cleared.append(.{ .string = category });

        var cat_obj: ObjectMap = .empty;
        try cat_obj.put(allocator, "wamr_samples", .{ .integer = wamr_s });
        try cat_obj.put(allocator, "wasmtime_samples", .{ .integer = wt_s });
        try cat_obj.put(allocator, "wamr_percent_of_run", .{ .float = wamr_share });
        try cat_obj.put(allocator, "wasmtime_percent_of_run", .{ .float = wasmtime_share });
        try cat_obj.put(allocator, "observed_delta_percentage_points", .{ .float = wamr_share - wasmtime_share });
        try cat_obj.put(
            allocator,
            "wasmtime_unknown_mixed_instruction_unresolved_or_global_unattributed_upper_percentage_points",
            .{ .float = wasmtime_unknown_upper },
        );
        try cat_obj.put(allocator, "wasmtime_instruction_unresolved_samples", .{ .integer = wasmtime_instruction_unresolved });
        try cat_obj.put(allocator, "wasmtime_global_unattributed_samples", .{ .integer = wasmtime_global_unattributed });
        try cat_obj.put(allocator, "conservative_headroom_percentage_points", .{ .float = conservative_headroom });
        try cat_obj.put(allocator, "clears_threshold", .{ .bool = clears });
        try categories.put(allocator, category, .{ .object = cat_obj });
    }

    const has_cleared = cleared.items.len > 0;
    var out: ObjectMap = .empty;
    try out.put(allocator, "threshold_percentage_points", .{ .float = threshold_pct });
    try out.put(allocator, "status", .{ .string = if (has_cleared) "cleared" else "not-cleared" });
    try out.put(allocator, "optimization_authorized", .{ .bool = has_cleared });
    try out.put(allocator, "cleared_categories", .{ .array = cleared });
    try out.put(
        allocator,
        "reason",
        .{ .string = if (has_cleared)
            "at least one shared proven category clears the conservative run-wide threshold"
        else
            "no shared proven category clears the conservative run-wide threshold after treating Wasmtime common-universe unknown/mixed, instruction-unresolved, and globally unattributed samples as a possible upper bound" },
    );
    try out.put(allocator, "categories", .{ .object = categories });
    return .{ .object = out };
}

fn primaryUnknownReasonCluster(
    allocator: Allocator,
    item: ObjectMap,
    diag: *Diagnostic,
) ProvenanceError![]const u8 {
    const cat_val = item.get("category") orelse Value.null;
    const category = if (cat_val == .string) cat_val.string else "";
    if (std.mem.eql(u8, category, "mixed")) return "mixed_categories";
    if (!std.mem.eql(u8, category, "unknown")) {
        return diag.fail(
            allocator,
            "expected unknown or mixed category for cluster classification, got {s}",
            .{category},
        );
    }
    const r_val = item.get("reason") orelse Value.null;
    const reason = if (r_val == .string) r_val.string else "";
    if (std.mem.indexOf(u8, reason, "control semantics are not proven bounds/check or algorithmic data") != null) {
        return "non_trap_control";
    }
    if (std.mem.indexOf(u8, reason, "consumer has multiple reaching definitions at a CFG join") != null) {
        return "ambiguous_cfg_join";
    }
    if (std.mem.indexOf(u8, reason, "cyclic producer-consumer path") != null or
        std.mem.indexOf(u8, reason, "value participates in cyclic data flow") != null)
    {
        return "cyclic_flow";
    }
    if (std.mem.indexOf(u8, reason, "value escapes through untyped memory or a signatureless return") != null) {
        return "untyped_escape";
    }
    if (std.mem.indexOf(u8, reason, "value reaches an opaque instruction or call boundary") != null or
        std.mem.indexOf(u8, reason, "instruction semantics are not recognized conservatively") != null)
    {
        return "opaque_or_call_boundary";
    }
    if (std.mem.indexOf(u8, reason, "combined memory and value-update costs cannot be isolated as pure ALU") != null) {
        return "combined_memory_writeback";
    }
    if (std.mem.indexOf(u8, reason, "definition has no proven consumer") != null or
        std.mem.indexOf(u8, reason, "value transformer has no proven live output") != null or
        std.mem.indexOf(u8, reason, "no terminal semantic consumer was proven") != null or
        std.mem.indexOf(u8, reason, "instruction is unreachable in the recovered CFG") != null)
    {
        return "dead_or_unconsumed";
    }
    return diag.fail(allocator, "unrecognized unknown provenance reason: {s}", .{reason});
}

pub fn classifyUnknownReasonClusters(
    allocator: Allocator,
    analysis_val: Value,
    name: []const u8,
    diag: *Diagnostic,
) ProvenanceError!Value {
    try validateAnalysisSamples(allocator, analysis_val, name, diag);
    const analysis = analysis_val.object;
    const total = analysis.get("total_run_samples").?.integer;
    const common = analysis.get("common_gating_universe").?.object;
    const common_categories = common.get("categories").?.object;
    const si_val = common.get("sampled_instructions") orelse {
        return diag.fail(allocator, "{s} common_gating_universe is missing sampled_instructions", .{name});
    };
    if (si_val != .array) {
        return diag.fail(allocator, "{s} common_gating_universe is missing sampled_instructions", .{name});
    }

    var expected_proven_samples: i64 = 0;
    for (PROVEN_CATEGORIES) |cat| {
        expected_proven_samples += common_categories.get(cat).?.object.get("samples").?.integer;
    }
    const expected_uncertain_samples = common_categories.get("unknown").?.object.get("samples").?.integer +
        common_categories.get("mixed").?.object.get("samples").?.integer;

    var seen_offsets: std.AutoArrayHashMapUnmanaged(i64, void) = .empty;
    var observed_proven_samples: i64 = 0;
    var cluster_samples: std.StringArrayHashMapUnmanaged(i64) = .empty;
    const ClusterRow = struct {
        offset: i64,
        samples: i64,
        val: Value,
    };
    var cluster_rows: std.StringArrayHashMapUnmanaged(std.ArrayList(ClusterRow)) = .empty;
    for (UNKNOWN_REASON_CLUSTERS) |cl| {
        try cluster_samples.put(allocator, cl, 0);
        try cluster_rows.put(allocator, cl, .empty);
    }

    for (si_val.array.items) |item_val| {
        if (item_val != .object) {
            return diag.fail(allocator, "{s} common_gating_universe.sampled_instructions contains a non-object row", .{name});
        }
        const item = item_val.object;
        const off_val = item.get("offset") orelse Value.null;
        if (off_val != .integer or off_val.integer < 0) {
            return diag.fail(allocator, "{s} sampled_instructions contains an invalid offset", .{name});
        }
        const offset = off_val.integer;
        if (seen_offsets.contains(offset)) {
            return diag.fail(allocator, "{s} sampled_instructions contains duplicate offset {d}", .{ name, offset });
        }
        try seen_offsets.put(allocator, offset, {});

        const s_val = item.get("samples") orelse Value.null;
        if (s_val != .integer or s_val.integer <= 0) {
            return diag.fail(allocator, "{s} sampled_instructions[{d}] must have positive integer samples", .{ name, offset });
        }
        const samples = s_val.integer;
        const cat_val = item.get("category") orelse Value.null;
        const category = if (cat_val == .string) cat_val.string else "";
        if (inList(&PROVEN_CATEGORIES, category)) {
            observed_proven_samples += samples;
            continue;
        }
        const cluster = try primaryUnknownReasonCluster(allocator, item, diag);
        cluster_samples.getPtr(cluster).?.* += samples;
        try cluster_rows.getPtr(cluster).?.append(allocator, .{
            .offset = offset,
            .samples = samples,
            .val = item_val,
        });
    }

    if (observed_proven_samples != expected_proven_samples) {
        return diag.fail(
            allocator,
            "{s} sampled_instructions proven samples ({d}) do not match common_gating_universe proven category sum ({d})",
            .{ name, observed_proven_samples, expected_proven_samples },
        );
    }
    var observed_uncertain_samples: i64 = 0;
    for (UNKNOWN_REASON_CLUSTERS) |cl| observed_uncertain_samples += cluster_samples.get(cl).?;
    if (observed_uncertain_samples != expected_uncertain_samples) {
        return diag.fail(
            allocator,
            "{s} sampled_instructions uncertain samples ({d}) do not match common_gating_universe unknown+mixed samples ({d})",
            .{ name, observed_uncertain_samples, expected_uncertain_samples },
        );
    }

    var clusters_obj: ObjectMap = .empty;
    for (UNKNOWN_REASON_CLUSTERS) |cluster| {
        const rows = cluster_rows.getPtr(cluster).?;
        std.mem.sort(ClusterRow, rows.items, {}, struct {
            fn lessThan(_: void, a: ClusterRow, b: ClusterRow) bool {
                if (a.samples != b.samples) return a.samples > b.samples;
                return a.offset < b.offset;
            }
        }.lessThan);
        var rows_arr = Array.init(allocator);
        for (rows.items) |r| try rows_arr.append(r.val);
        const samples = cluster_samples.get(cluster).?;

        var cl_obj: ObjectMap = .empty;
        try cl_obj.put(allocator, "samples", .{ .integer = samples });
        try cl_obj.put(
            allocator,
            "percent_of_run",
            .{ .float = 100.0 * @as(f64, @floatFromInt(samples)) / @as(f64, @floatFromInt(total)) },
        );
        try cl_obj.put(
            allocator,
            "percent_of_uncertain",
            .{ .float = if (expected_uncertain_samples > 0)
                100.0 * @as(f64, @floatFromInt(samples)) / @as(f64, @floatFromInt(expected_uncertain_samples))
            else
                0.0 },
        );
        try cl_obj.put(allocator, "sampled_instruction_count", .{ .integer = @intCast(rows.items.len) });
        try cl_obj.put(allocator, "sampled_instructions", .{ .array = rows_arr });
        try clusters_obj.put(allocator, cluster, .{ .object = cl_obj });
    }

    var out: ObjectMap = .empty;
    try out.put(allocator, "uncertain_samples", .{ .integer = expected_uncertain_samples });
    try out.put(
        allocator,
        "uncertain_percent_of_run",
        .{ .float = 100.0 * @as(f64, @floatFromInt(expected_uncertain_samples)) / @as(f64, @floatFromInt(total)) },
    );
    try out.put(allocator, "clusters", .{ .object = clusters_obj });
    return .{ .object = out };
}

fn traceTerminalSinksUnblocked(
    allocator: Allocator,
    index: usize,
    instructions: []const Instruction,
    effects: []const Effect,
    consumers: ConsumersMap,
    reachable: IndexSet,
    structural_guard_branches: IndexSet,
    structural_guard_definitions: DefinitionSet,
) !struct { sink_class: []const u8, evidence: []const []const u8 } {
    const inst = instructions[index];
    const effect = effects[index];
    if (!reachable.contains(index)) {
        const ev = try allocator.dupe([]const u8, &.{
            try std.fmt.allocPrint(allocator, "unreachable:{s}@{d}", .{ inst.text, inst.offset }),
        });
        return .{ .sink_class = "dead_or_unconsumed", .evidence = ev };
    }
    if (!effect.recognized) {
        const ev = try allocator.dupe([]const u8, &.{
            try std.fmt.allocPrint(allocator, "unrecognized:{s}@{d}", .{ inst.text, inst.offset }),
        });
        return .{ .sink_class = "opaque_or_call_boundary", .evidence = ev };
    }
    if (effect.memory) {
        const ev = try allocator.dupe([]const u8, &.{
            try std.fmt.allocPrint(allocator, "memory:{s}@{d}", .{ inst.text, inst.offset }),
        });
        return .{ .sink_class = "combined_memory_writeback", .evidence = ev };
    }

    var definitions: std.StringArrayHashMapUnmanaged(void) = .empty;
    for ([_]std.StringArrayHashMapUnmanaged(void){ effect.primary_definitions, effect.secondary_definitions }) |set| {
        var it = set.iterator();
        while (it.next()) |e| {
            const reg = e.key_ptr.*;
            if (consumers.get(.{ .instruction = @intCast(index), .register = reg })) |c_list| {
                if (c_list.items.len > 0) try definitions.put(allocator, reg, {});
            }
        }
    }
    if (definitions.count() == 0) {
        var p_it = effect.primary_definitions.iterator();
        while (p_it.next()) |e| try definitions.put(allocator, e.key_ptr.*, {});
    }
    if (definitions.count() == 0) {
        const ev = try allocator.dupe([]const u8, &.{
            try std.fmt.allocPrint(allocator, "no_defs:{s}@{d}", .{ inst.text, inst.offset }),
        });
        return .{ .sink_class = "dead_or_unconsumed", .evidence = ev };
    }

    var def_regs: std.ArrayList([]const u8) = .empty;
    var d_it = definitions.iterator();
    while (d_it.next()) |e| try def_regs.append(allocator, e.key_ptr.*);
    std.mem.sort([]const u8, def_regs.items, {}, struct {
        fn lessThan(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.order(u8, a, b) == .lt;
        }
    }.lessThan);

    var worklist: std.ArrayList(Definition) = .empty;
    for (def_regs.items) |reg| {
        try worklist.append(allocator, .{ .instruction = @intCast(index), .register = reg });
    }

    var visited: DefinitionSet = .empty;
    var terminals: std.StringArrayHashMapUnmanaged(void) = .empty;
    var evidence: std.ArrayList([]const u8) = .empty;

    const record = struct {
        fn call(
            alloc: Allocator,
            terms: *std.StringArrayHashMapUnmanaged(void),
            evs: *std.ArrayList([]const u8),
            kind: []const u8,
            detail: []const u8,
        ) !void {
            try terms.put(alloc, kind, {});
            if (!inList(evs.items, detail)) {
                try evs.append(alloc, detail);
            }
        }
    }.call;

    while (worklist.items.len > 0) {
        const definition = worklist.pop().?;
        if (visited.contains(definition)) continue;
        try visited.put(allocator, definition, {});
        const uses_opt = consumers.get(definition);
        if (uses_opt == null or uses_opt.?.items.len == 0) {
            if (definition.instruction == @as(i64, @intCast(index))) {
                try record(
                    allocator,
                    &terminals,
                    &evidence,
                    "dead",
                    try std.fmt.allocPrint(allocator, "dead:{s}@{d}", .{ inst.text, inst.offset }),
                );
            }
            continue;
        }
        for (uses_opt.?.items) |consumer| {
            const use_inst = instructions[consumer.instruction];
            const role = consumer.use.role;
            if (std.mem.eql(u8, role, "address")) {
                try record(
                    allocator,
                    &terminals,
                    &evidence,
                    "address",
                    try std.fmt.allocPrint(allocator, "address:{s}@{d}", .{ use_inst.text, use_inst.offset }),
                );
            } else if (std.mem.eql(u8, role, "escape")) {
                try record(
                    allocator,
                    &terminals,
                    &evidence,
                    "escape",
                    try std.fmt.allocPrint(allocator, "escape:{s}@{d}", .{ use_inst.text, use_inst.offset }),
                );
            } else if (std.mem.eql(u8, role, "control")) {
                if (structural_guard_branches.contains(consumer.instruction)) {
                    try record(
                        allocator,
                        &terminals,
                        &evidence,
                        "structural_guard",
                        try std.fmt.allocPrint(allocator, "structural_guard:{s}@{d}", .{ use_inst.text, use_inst.offset }),
                    );
                } else {
                    try record(
                        allocator,
                        &terminals,
                        &evidence,
                        "non_trap_control",
                        try std.fmt.allocPrint(allocator, "control:{s}@{d}", .{ use_inst.text, use_inst.offset }),
                    );
                }
            } else if (std.mem.eql(u8, role, "compare")) {
                const nzcv_def: Definition = .{
                    .instruction = @intCast(consumer.instruction),
                    .register = "nzcv",
                };
                if (structural_guard_definitions.contains(nzcv_def)) {
                    try record(
                        allocator,
                        &terminals,
                        &evidence,
                        "structural_guard",
                        try std.fmt.allocPrint(allocator, "structural_guard:{s}@{d}", .{ use_inst.text, use_inst.offset }),
                    );
                } else {
                    const nzcv_uses = consumers.get(nzcv_def);
                    if (nzcv_uses == null or nzcv_uses.?.items.len == 0) {
                        try record(
                            allocator,
                            &terminals,
                            &evidence,
                            "non_trap_control",
                            try std.fmt.allocPrint(allocator, "compare:{s}@{d}", .{ use_inst.text, use_inst.offset }),
                        );
                    } else {
                        try worklist.append(allocator, nzcv_def);
                    }
                }
            } else if (std.mem.eql(u8, role, "data")) {
                try record(
                    allocator,
                    &terminals,
                    &evidence,
                    "algorithmic",
                    try std.fmt.allocPrint(allocator, "data:{s}@{d}", .{ use_inst.text, use_inst.offset }),
                );
            } else if (std.mem.eql(u8, role, "unknown")) {
                try record(
                    allocator,
                    &terminals,
                    &evidence,
                    "opaque",
                    try std.fmt.allocPrint(allocator, "opaque:{s}@{d}", .{ use_inst.text, use_inst.offset }),
                );
            } else if (!std.mem.eql(u8, role, "value")) {
                try record(
                    allocator,
                    &terminals,
                    &evidence,
                    "opaque",
                    try std.fmt.allocPrint(allocator, "unrecognized_role:{s}@{d}", .{ role, use_inst.offset }),
                );
            }

            var propagation_targets: std.ArrayList([]const u8) = .empty;
            for (consumer.use.propagate_to) |target_register| {
                if (effects[consumer.instruction].definitions.contains(target_register)) {
                    try propagation_targets.append(allocator, target_register);
                }
            }
            if (propagation_targets.items.len != consumer.use.propagate_to.len) {
                try record(
                    allocator,
                    &terminals,
                    &evidence,
                    "opaque",
                    try std.fmt.allocPrint(allocator, "missing_target:{s}@{d}", .{ use_inst.text, use_inst.offset }),
                );
            }
            var active_targets: std.ArrayList([]const u8) = .empty;
            for (propagation_targets.items) |target_register| {
                const td: Definition = .{
                    .instruction = @intCast(consumer.instruction),
                    .register = target_register,
                };
                if (consumers.get(td)) |c_list| {
                    if (c_list.items.len > 0) try active_targets.append(allocator, target_register);
                }
            }
            if (propagation_targets.items.len > 0 and active_targets.items.len == 0) {
                try record(
                    allocator,
                    &terminals,
                    &evidence,
                    "dead",
                    try std.fmt.allocPrint(allocator, "dead_transform:{s}@{d}", .{ use_inst.text, use_inst.offset }),
                );
            }
            for (active_targets.items) |target_register| {
                try worklist.append(allocator, .{
                    .instruction = @intCast(consumer.instruction),
                    .register = target_register,
                });
            }
        }
    }

    const ev_slice = try evidence.toOwnedSlice(allocator);
    if (terminals.count() == 0 or (terminals.count() == 1 and terminals.contains("dead"))) {
        return .{ .sink_class = "dead_or_unconsumed", .evidence = ev_slice };
    }
    var live: std.StringArrayHashMapUnmanaged(void) = .empty;
    var t_it = terminals.iterator();
    while (t_it.next()) |e| {
        if (!std.mem.eql(u8, e.key_ptr.*, "dead")) {
            try live.put(allocator, e.key_ptr.*, {});
        }
    }
    if (live.count() == 1) {
        if (live.contains("address")) return .{ .sink_class = "pure_address", .evidence = ev_slice };
        if (live.contains("structural_guard")) return .{ .sink_class = "pure_structural_guard", .evidence = ev_slice };
        if (live.contains("algorithmic")) return .{ .sink_class = "pure_algorithmic", .evidence = ev_slice };
        if (live.contains("non_trap_control")) return .{ .sink_class = "non_trap_control", .evidence = ev_slice };
        if (live.contains("escape")) return .{ .sink_class = "untyped_escape", .evidence = ev_slice };
    }
    if (live.contains("address")) return .{ .sink_class = "mixed_address_and_control_or_escape", .evidence = ev_slice };
    if (live.contains("opaque")) return .{ .sink_class = "opaque_or_call_boundary", .evidence = ev_slice };
    if (live.contains("non_trap_control")) return .{ .sink_class = "non_trap_control", .evidence = ev_slice };
    if (live.contains("escape")) return .{ .sink_class = "untyped_escape", .evidence = ev_slice };
    return .{ .sink_class = "opaque_or_call_boundary", .evidence = ev_slice };
}

pub fn analyzeTerminalSinkExhaustion(
    allocator: Allocator,
    instructions: []const InstructionInput,
    options: AnalyzeOptions,
    diag: *Diagnostic,
) ProvenanceError!Value {
    const primary = try analyzeInstructionStream(allocator, instructions, options, diag);
    const normalized = try normalizeInstructions(allocator, instructions, options.address_base, diag);
    const effects = try allocator.alloc(Effect, normalized.len);
    for (normalized, 0..) |inst, idx| {
        effects[idx] = try instructionEffect(allocator, inst);
    }
    const cfg = try buildCfg(allocator, normalized, effects);
    const reachable = try computeReachable(allocator, cfg.successors);
    const reaching = try reachingDefinitions(allocator, effects, cfg.predecessors, reachable);
    const dominators = try computeDominators(allocator, cfg.predecessors, reachable);
    const guards = try structuralAddressGuardBranches(
        allocator,
        effects,
        cfg.successors,
        reaching.incoming,
        dominators,
        reachable,
    );
    const consumers = try buildConsumers(allocator, effects, reaching.incoming, reachable);
    var memo: TraceMemo = .empty;

    var samples_map: std.AutoArrayHashMapUnmanaged(i64, i64) = .empty;
    for (options.samples_by_offset) |entry| {
        try samples_map.put(allocator, entry.offset, entry.samples);
    }

    var common_indices: std.ArrayList(usize) = .empty;
    for (options.broad_classes, 0..) |broad_class, idx| {
        if (std.mem.eql(u8, broad_class, "alu") or effects[idx].gating_candidate or !effects[idx].recognized) {
            try common_indices.append(allocator, idx);
        }
    }

    var primary_by_offset: std.AutoArrayHashMapUnmanaged(i64, Value) = .empty;
    const primary_sampled = primary.object.get("common_gating_universe").?.object.get("sampled_instructions").?.array.items;
    for (primary_sampled) |row| {
        try primary_by_offset.put(allocator, row.object.get("offset").?.integer, row);
    }

    const RowItem = struct {
        offset: i64,
        samples: i64,
        val: Value,
    };
    var sink_samples: std.StringArrayHashMapUnmanaged(i64) = .empty;
    var sink_static: std.StringArrayHashMapUnmanaged(i64) = .empty;
    var sink_rows: std.StringArrayHashMapUnmanaged(std.ArrayList(RowItem)) = .empty;
    for (TERMINAL_SINK_CLASSES) |sink| {
        try sink_samples.put(allocator, sink, 0);
        try sink_static.put(allocator, sink, 0);
        try sink_rows.put(allocator, sink, .empty);
    }

    const reclass_keys = [_]struct { sink: []const u8, strict: []const u8 }{
        .{ .sink = "pure_address", .strict = "address_generation" },
        .{ .sink = "pure_structural_guard", .strict = "structural_address_guard" },
        .{ .sink = "pure_algorithmic", .strict = "algorithmic_alu" },
    };
    var reclassified_samples: std.StringArrayHashMapUnmanaged(i64) = .empty;
    var reclassified_static: std.StringArrayHashMapUnmanaged(i64) = .empty;
    var reclassified_rows: std.StringArrayHashMapUnmanaged(std.ArrayList(RowItem)) = .empty;
    for (reclass_keys) |rk| {
        try reclassified_samples.put(allocator, rk.sink, 0);
        try reclassified_static.put(allocator, rk.sink, 0);
        try reclassified_rows.put(allocator, rk.sink, .empty);
    }

    for (common_indices.items) |idx| {
        const inst = normalized[idx];
        const offset = inst.offset;
        const samples = samples_map.get(offset) orelse 0;
        const unblocked = try traceTerminalSinksUnblocked(
            allocator,
            idx,
            normalized,
            effects,
            consumers,
            reachable,
            guards.guard_branches,
            guards.guard_definitions,
        );

        var strict_cat: []const u8 = "unknown";
        var strict_reason: []const u8 = "";
        const effect = effects[idx];
        if (!reachable.contains(idx)) {
            strict_reason = "instruction is unreachable in the recovered CFG";
        } else if (!effect.recognized) {
            strict_reason = "instruction semantics are not recognized conservatively";
        } else if (effect.memory) {
            strict_reason = "combined memory and value-update costs cannot be isolated as pure ALU";
        } else {
            var definitions: std.StringArrayHashMapUnmanaged(void) = .empty;
            for ([_]std.StringArrayHashMapUnmanaged(void){ effect.primary_definitions, effect.secondary_definitions }) |set| {
                var it = set.iterator();
                while (it.next()) |e| {
                    const reg = e.key_ptr.*;
                    if (consumers.get(.{ .instruction = @intCast(idx), .register = reg })) |c_list| {
                        if (c_list.items.len > 0) try definitions.put(allocator, reg, {});
                    }
                }
            }
            if (definitions.count() == 0) {
                var p_it = effect.primary_definitions.iterator();
                while (p_it.next()) |e| try definitions.put(allocator, e.key_ptr.*, {});
            }
            var combined: Trace = .{};
            var d_it = definitions.iterator();
            while (d_it.next()) |e| {
                const sub = try traceDefinition(
                    allocator,
                    .{ .instruction = @intCast(idx), .register = e.key_ptr.* },
                    normalized,
                    effects,
                    consumers,
                    guards.guard_branches,
                    guards.guard_definitions,
                    .empty,
                    &memo,
                );
                try combined.merge(allocator, sub);
            }
            const sc = try classifyTraceCategory(allocator, combined);
            strict_cat = sc.category;
            strict_reason = sc.reason;
        }

        sink_samples.getPtr(unblocked.sink_class).?.* += samples;
        sink_static.getPtr(unblocked.sink_class).?.* += 1;

        var ev_arr = Array.init(allocator);
        const ev_limit = @min(unblocked.evidence.len, 12);
        for (unblocked.evidence[0..ev_limit]) |ev_str| {
            try ev_arr.append(.{ .string = ev_str });
        }

        var row_obj: ObjectMap = .empty;
        try row_obj.put(allocator, "offset", .{ .integer = offset });
        try row_obj.put(allocator, "address", .{ .integer = inst.address });
        try row_obj.put(allocator, "instruction", .{ .string = inst.text });
        try row_obj.put(allocator, "samples", .{ .integer = samples });
        try row_obj.put(
            allocator,
            "percent_of_run",
            .{ .float = 100.0 * @as(f64, @floatFromInt(samples)) / @as(f64, @floatFromInt(options.total_run_samples)) },
        );
        try row_obj.put(allocator, "broad_class", .{ .string = options.broad_classes[idx] });
        try row_obj.put(allocator, "strict_category", .{ .string = strict_cat });
        try row_obj.put(allocator, "strict_reason", .{ .string = strict_reason });
        try row_obj.put(allocator, "terminal_sink_class", .{ .string = unblocked.sink_class });
        try row_obj.put(allocator, "terminal_evidence", .{ .array = ev_arr });
        const row_val: Value = .{ .object = row_obj };

        if (samples > 0) {
            try sink_rows.getPtr(unblocked.sink_class).?.append(allocator, .{
                .offset = offset,
                .samples = samples,
                .val = row_val,
            });
        }

        for (reclass_keys) |rk| {
            if (std.mem.eql(u8, unblocked.sink_class, rk.sink)) {
                if (!std.mem.eql(u8, strict_cat, rk.strict)) {
                    reclassified_static.getPtr(rk.sink).?.* += 1;
                    reclassified_samples.getPtr(rk.sink).?.* += samples;
                    if (samples > 0) {
                        const chosen = primary_by_offset.get(offset) orelse row_val;
                        try reclassified_rows.getPtr(rk.sink).?.append(allocator, .{
                            .offset = offset,
                            .samples = samples,
                            .val = chosen,
                        });
                    }
                }
            }
        }
    }

    const common = primary.object.get("common_gating_universe").?.object;
    const common_samples = common.get("samples").?.integer;
    var sum_sinks: i64 = 0;
    for (TERMINAL_SINK_CLASSES) |sink| sum_sinks += sink_samples.get(sink).?;
    if (sum_sinks != common_samples) {
        return diag.fail(
            allocator,
            "terminal sink exhaustion sample sum does not match common_gating_universe.samples",
            .{},
        );
    }
    for (reclass_keys) |rk| {
        const strict_val = common.get("categories").?.object.get(rk.strict).?.object.get("samples").?.integer;
        const unblocked_val = sink_samples.get(rk.sink).?;
        if (unblocked_val < strict_val) {
            return diag.fail(
                allocator,
                "unblocked {s} samples ({d}) cannot be less than strict {s} samples ({d})",
                .{ rk.sink, unblocked_val, rk.strict, strict_val },
            );
        }
    }

    var sink_classes_out: ObjectMap = .empty;
    for (TERMINAL_SINK_CLASSES) |sink| {
        const rows = sink_rows.getPtr(sink).?;
        std.mem.sort(RowItem, rows.items, {}, struct {
            fn lessThan(_: void, a: RowItem, b: RowItem) bool {
                if (a.samples != b.samples) return a.samples > b.samples;
                return a.offset < b.offset;
            }
        }.lessThan);
        var rows_arr = Array.init(allocator);
        for (rows.items) |r| try rows_arr.append(r.val);
        const s_val = sink_samples.get(sink).?;

        var s_obj: ObjectMap = .empty;
        try s_obj.put(allocator, "samples", .{ .integer = s_val });
        try s_obj.put(
            allocator,
            "percent_of_run",
            .{ .float = 100.0 * @as(f64, @floatFromInt(s_val)) / @as(f64, @floatFromInt(options.total_run_samples)) },
        );
        try s_obj.put(
            allocator,
            "percent_of_common_universe",
            .{ .float = if (common_samples > 0)
                100.0 * @as(f64, @floatFromInt(s_val)) / @as(f64, @floatFromInt(common_samples))
            else
                0.0 },
        );
        try s_obj.put(allocator, "static_instructions", .{ .integer = sink_static.get(sink).? });
        try s_obj.put(allocator, "sampled_instructions", .{ .array = rows_arr });
        try sink_classes_out.put(allocator, sink, .{ .object = s_obj });
    }

    var reclassified_out: ObjectMap = .empty;
    for (reclass_keys) |rk| {
        const rows = reclassified_rows.getPtr(rk.sink).?;
        std.mem.sort(RowItem, rows.items, {}, struct {
            fn lessThan(_: void, a: RowItem, b: RowItem) bool {
                if (a.samples != b.samples) return a.samples > b.samples;
                return a.offset < b.offset;
            }
        }.lessThan);
        var rows_arr = Array.init(allocator);
        for (rows.items) |r| try rows_arr.append(r.val);
        const s_val = reclassified_samples.get(rk.sink).?;

        var r_obj: ObjectMap = .empty;
        try r_obj.put(allocator, "samples", .{ .integer = s_val });
        try r_obj.put(
            allocator,
            "percent_of_run",
            .{ .float = 100.0 * @as(f64, @floatFromInt(s_val)) / @as(f64, @floatFromInt(options.total_run_samples)) },
        );
        try r_obj.put(allocator, "static_instructions", .{ .integer = reclassified_static.get(rk.sink).? });
        try r_obj.put(allocator, "sampled_instructions", .{ .array = rows_arr });
        try reclassified_out.put(allocator, rk.sink, .{ .object = r_obj });
    }

    var out: ObjectMap = .empty;
    try out.put(allocator, "total_run_samples", .{ .integer = options.total_run_samples });
    try out.put(allocator, "common_samples", .{ .integer = common_samples });
    try out.put(allocator, "common_static_instructions", .{ .integer = @intCast(common_indices.items.len) });
    try out.put(allocator, "sink_classes", .{ .object = sink_classes_out });
    try out.put(allocator, "reclassified_from_unknown", .{ .object = reclassified_out });
    try out.put(allocator, "primary_analysis", primary);
    return .{ .object = out };
}

pub fn validateTerminalSinkExhaustion(
    allocator: Allocator,
    analysis_val: Value,
    exhaustion_val: Value,
    name: []const u8,
    diag: *Diagnostic,
) ProvenanceError!void {
    try validateAnalysisSamples(
        allocator,
        analysis_val,
        try std.fmt.allocPrint(allocator, "{s}.analysis", .{name}),
        diag,
    );
    if (exhaustion_val != .object) {
        return diag.fail(allocator, "{s} must be a mapping", .{name});
    }
    const analysis = analysis_val.object;
    const exhaustion = exhaustion_val.object;
    const total = analysis.get("total_run_samples").?.integer;
    const common = analysis.get("common_gating_universe").?.object;
    const common_samples = common.get("samples").?.integer;

    const ex_total = exhaustion.get("total_run_samples") orelse Value.null;
    if (ex_total != .integer or ex_total.integer != total) {
        return diag.fail(allocator, "{s} total_run_samples does not match primary analysis", .{name});
    }
    const ex_common = exhaustion.get("common_samples") orelse Value.null;
    if (ex_common != .integer or ex_common.integer != common_samples) {
        return diag.fail(allocator, "{s} common_samples does not match common_gating_universe.samples", .{name});
    }
    const sc_val = exhaustion.get("sink_classes") orelse {
        return diag.fail(allocator, "{s} is missing sink_classes", .{name});
    };
    if (sc_val != .object) {
        return diag.fail(allocator, "{s} is missing sink_classes", .{name});
    }
    const sink_classes = sc_val.object;
    var total_sink_samples: i64 = 0;
    for (TERMINAL_SINK_CLASSES) |sink| {
        const s_entry = sink_classes.get(sink) orelse {
            return diag.fail(allocator, "{s}.sink_classes is missing {s}", .{ name, sink });
        };
        if (s_entry != .object) {
            return diag.fail(allocator, "{s}.sink_classes is missing {s}", .{ name, sink });
        }
        const s_val = s_entry.object.get("samples") orelse Value.null;
        if (s_val != .integer or s_val.integer < 0) {
            return diag.fail(allocator, "{s}.sink_classes.{s}.samples must be a non-negative integer", .{ name, sink });
        }
        total_sink_samples += s_val.integer;
    }
    if (total_sink_samples != common_samples) {
        return diag.fail(
            allocator,
            "{s}.sink_classes sample sum ({d}) does not equal common_samples ({d})",
            .{ name, total_sink_samples, common_samples },
        );
    }

    const rc_val = exhaustion.get("reclassified_from_unknown") orelse {
        return diag.fail(allocator, "{s} is missing reclassified_from_unknown", .{name});
    };
    if (rc_val != .object) {
        return diag.fail(allocator, "{s} is missing reclassified_from_unknown", .{name});
    }
    const reclassified = rc_val.object;

    for ([_]struct { sink: []const u8, strict: []const u8 }{
        .{ .sink = "pure_address", .strict = "address_generation" },
        .{ .sink = "pure_structural_guard", .strict = "structural_address_guard" },
        .{ .sink = "pure_algorithmic", .strict = "algorithmic_alu" },
    }) |pair| {
        const strict_samples = common.get("categories").?.object.get(pair.strict).?.object.get("samples").?.integer;
        const unblocked_samples = sink_classes.get(pair.sink).?.object.get("samples").?.integer;
        if (unblocked_samples < strict_samples) {
            return diag.fail(
                allocator,
                "{s} {s} samples ({d}) are less than strict proven samples ({d})",
                .{ name, pair.sink, unblocked_samples, strict_samples },
            );
        }
        const entry = reclassified.get(pair.sink) orelse {
            return diag.fail(allocator, "{s}.reclassified_from_unknown is missing {s}", .{ name, pair.sink });
        };
        if (entry != .object) {
            return diag.fail(allocator, "{s}.reclassified_from_unknown is missing {s}", .{ name, pair.sink });
        }
        const reclass_samples = entry.object.get("samples") orelse Value.null;
        if (reclass_samples != .integer or reclass_samples.integer != unblocked_samples - strict_samples) {
            return diag.fail(
                allocator,
                "{s}.reclassified_from_unknown.{s}.samples does not equal unblocked - strict delta ({d})",
                .{ name, pair.sink, unblocked_samples - strict_samples },
            );
        }
    }
}

pub fn compareTerminalSinkExhaustion(
    allocator: Allocator,
    wamr_val: Value,
    wasmtime_val: Value,
    wasmtime_exhaustion_val: Value,
    wamr_exhaustion_val: ?Value,
    threshold_pct: f64,
    diag: *Diagnostic,
) ProvenanceError!Value {
    const strict_gate = try compareEngineAnalyses(allocator, wamr_val, wasmtime_val, threshold_pct, diag);
    try validateTerminalSinkExhaustion(allocator, wasmtime_val, wasmtime_exhaustion_val, "wasmtime_exhaustion", diag);
    if (wamr_exhaustion_val) |we| {
        try validateTerminalSinkExhaustion(allocator, wamr_val, we, "wamr_exhaustion", diag);
    }

    const wamr = wamr_val.object;
    const wasmtime = wasmtime_val.object;
    const wasmtime_exhaustion = wasmtime_exhaustion_val.object;

    const wasmtime_total = wasmtime.get("total_run_samples").?.integer;
    const wamr_total = wamr.get("total_run_samples").?.integer;
    const wasmtime_common = wasmtime.get("common_gating_universe").?.object.get("samples").?.integer;
    const wasmtime_instruction_unresolved = if (wasmtime.get("sample_mapping")) |sm|
        if (sm == .object and sm.object.get("unresolved_function_samples") != null)
            sm.object.get("unresolved_function_samples").?.integer
        else
            0
    else
        0;
    const wasmtime_global_unattributed = wasmtime.get("global_sample_mapping").?.object.get("unattributed_samples").?.integer;

    const wt_sinks = wasmtime_exhaustion.get("sink_classes").?.object;
    const wt_pure_address = wt_sinks.get("pure_address").?.object.get("samples").?.integer;
    const wt_pure_guard = wt_sinks.get("pure_structural_guard").?.object.get("samples").?.integer;
    const wt_pure_algo = wt_sinks.get("pure_algorithmic").?.object.get("samples").?.integer;
    const wt_non_proven_common_samples = wasmtime_common - (wt_pure_address + wt_pure_guard + wt_pure_algo);

    var categories_out: ObjectMap = .empty;
    var cleared = Array.init(allocator);
    var all_exhausted = true;

    for (GATE_CATEGORIES) |category| {
        const sink_key = if (std.mem.eql(u8, category, "address_generation")) "pure_address" else "pure_algorithmic";
        const strict_wamr_samples = wamr.get("common_gating_universe").?.object
            .get("categories").?.object.get(category).?.object.get("samples").?.integer;
        const strict_wt_samples = wasmtime.get("common_gating_universe").?.object
            .get("categories").?.object.get(category).?.object.get("samples").?.integer;
        const unblocked_wamr_samples = if (wamr_exhaustion_val) |we|
            we.object.get("sink_classes").?.object.get(sink_key).?.object.get("samples").?.integer
        else
            strict_wamr_samples;
        const unblocked_wt_samples = if (std.mem.eql(u8, sink_key, "pure_address")) wt_pure_address else wt_pure_algo;
        const wt_reclassified_target = unblocked_wt_samples - strict_wt_samples;

        var wt_reclassified_disjoint: i64 = 0;
        for ([_]struct { sink: []const u8, cat: []const u8, unblocked: i64 }{
            .{ .sink = "pure_address", .cat = "address_generation", .unblocked = wt_pure_address },
            .{ .sink = "pure_structural_guard", .cat = "structural_address_guard", .unblocked = wt_pure_guard },
            .{ .sink = "pure_algorithmic", .cat = "algorithmic_alu", .unblocked = wt_pure_algo },
        }) |other| {
            if (!std.mem.eql(u8, other.sink, sink_key)) {
                const other_strict = wasmtime.get("common_gating_universe").?.object
                    .get("categories").?.object.get(other.cat).?.object.get("samples").?.integer;
                wt_reclassified_disjoint += other.unblocked - other_strict;
            }
        }

        const unblocked_wamr_share = 100.0 * @as(f64, @floatFromInt(unblocked_wamr_samples)) / @as(f64, @floatFromInt(wamr_total));
        const unblocked_wt_share = 100.0 * @as(f64, @floatFromInt(unblocked_wt_samples)) / @as(f64, @floatFromInt(wasmtime_total));
        const unblocked_wt_unknown_upper_samples = wt_non_proven_common_samples +
            wasmtime_instruction_unresolved +
            wasmtime_global_unattributed;
        const unblocked_wt_unknown_upper_pct = 100.0 *
            @as(f64, @floatFromInt(unblocked_wt_unknown_upper_samples)) /
            @as(f64, @floatFromInt(wasmtime_total));
        const unblocked_conservative_headroom = unblocked_wamr_share - (unblocked_wt_share + unblocked_wt_unknown_upper_pct);

        const wamr_mixed_samples: i64 = if (std.mem.eql(u8, category, "address_generation"))
            if (wamr_exhaustion_val) |we|
                we.object.get("sink_classes").?.object.get("mixed_address_and_control_or_escape").?.object.get("samples").?.integer
            else
                wamr.get("common_gating_universe").?.object.get("categories").?.object.get("mixed").?.object.get("samples").?.integer
        else
            0;

        const wamr_mixed_upper_share = 100.0 *
            @as(f64, @floatFromInt(unblocked_wamr_samples + wamr_mixed_samples)) /
            @as(f64, @floatFromInt(wamr_total));
        const wamr_mixed_credit_headroom = wamr_mixed_upper_share - (unblocked_wt_share + unblocked_wt_unknown_upper_pct);
        const clears = unblocked_conservative_headroom >= threshold_pct;
        const proven_exhausted = !clears and wamr_mixed_credit_headroom < threshold_pct;
        if (clears) try cleared.append(.{ .string = category });
        if (!proven_exhausted) all_exhausted = false;

        const strict_headroom = strict_gate.object.get("categories").?.object.get(category).?.object
            .get("conservative_headroom_percentage_points").?.float;

        var cat_obj: ObjectMap = .empty;
        try cat_obj.put(allocator, "strict_wamr_samples", .{ .integer = strict_wamr_samples });
        try cat_obj.put(allocator, "strict_wasmtime_samples", .{ .integer = strict_wt_samples });
        try cat_obj.put(allocator, "unblocked_wamr_samples", .{ .integer = unblocked_wamr_samples });
        try cat_obj.put(allocator, "unblocked_wasmtime_samples", .{ .integer = unblocked_wt_samples });
        try cat_obj.put(allocator, "wasmtime_reclassified_from_unknown_samples", .{ .integer = wt_reclassified_target });
        try cat_obj.put(allocator, "wasmtime_reclassified_to_disjoint_proven_samples", .{ .integer = wt_reclassified_disjoint });
        try cat_obj.put(allocator, "wasmtime_remaining_uncertain_common_samples", .{ .integer = wt_non_proven_common_samples });
        try cat_obj.put(allocator, "wasmtime_remaining_upper_bound_samples", .{ .integer = unblocked_wt_unknown_upper_samples });
        try cat_obj.put(allocator, "unblocked_wamr_percent_of_run", .{ .float = unblocked_wamr_share });
        try cat_obj.put(allocator, "unblocked_wasmtime_percent_of_run", .{ .float = unblocked_wt_share });
        try cat_obj.put(allocator, "unblocked_wasmtime_unknown_upper_percentage_points", .{ .float = unblocked_wt_unknown_upper_pct });
        try cat_obj.put(allocator, "strict_conservative_headroom_percentage_points", .{ .float = strict_headroom });
        try cat_obj.put(allocator, "unblocked_conservative_headroom_percentage_points", .{ .float = unblocked_conservative_headroom });
        try cat_obj.put(allocator, "wamr_mixed_credit_upper_bound_headroom_percentage_points", .{ .float = wamr_mixed_credit_headroom });
        try cat_obj.put(allocator, "clears_threshold", .{ .bool = clears });
        try cat_obj.put(allocator, "proven_exhausted", .{ .bool = proven_exhausted });
        try categories_out.put(allocator, category, .{ .object = cat_obj });
    }

    const has_cleared = cleared.items.len > 0;
    var out: ObjectMap = .empty;
    try out.put(allocator, "threshold_percentage_points", .{ .float = threshold_pct });
    try out.put(
        allocator,
        "status",
        .{ .string = if (has_cleared) "cleared" else if (all_exhausted) "exhausted" else "not-cleared" },
    );
    try out.put(allocator, "optimization_authorized", .{ .bool = has_cleared });
    try out.put(allocator, "proven_exhausted", .{ .bool = all_exhausted });
    try out.put(allocator, "cleared_categories", .{ .array = cleared });
    try out.put(allocator, "categories", .{ .object = categories_out });
    return .{ .object = out };
}

fn makeTestInst(allocator: Allocator, offset: i64, text: []const u8) !InstructionInput {
    const split = try splitInstructionText(allocator, text);
    return .{
        .offset = offset,
        .size = 4,
        .mnemonic = split.mnemonic,
        .operands = split.operands,
        .text = text,
        .address = offset,
    };
}

fn runTestAnalyze(
    allocator: Allocator,
    lines: []const []const u8,
    alu_offsets: []const i64,
    samples_opt: ?[]const SampleEntry,
    total: i64,
) !Value {
    var insts: std.ArrayList(InstructionInput) = .empty;
    var classes: std.ArrayList([]const u8) = .empty;
    for (lines, 0..) |line, idx| {
        const off: i64 = @intCast(idx * 4);
        try insts.append(allocator, try makeTestInst(allocator, off, line));
        var is_alu = false;
        for (alu_offsets) |ao| {
            if (ao == off) {
                is_alu = true;
                break;
            }
        }
        try classes.append(allocator, if (is_alu) "alu" else "other");
    }
    var default_samples: std.ArrayList(SampleEntry) = .empty;
    if (samples_opt == null) {
        for (alu_offsets) |ao| {
            try default_samples.append(allocator, .{ .offset = ao, .samples = 1 });
        }
    }
    var diag: Diagnostic = .{};
    return analyzeInstructionStream(allocator, insts.items, .{
        .broad_classes = classes.items,
        .samples_by_offset = samples_opt orelse default_samples.items,
        .total_run_samples = total,
    }, &diag);
}

fn getCategorySamples(result: Value, name: []const u8) i64 {
    return result.object.get("categories").?.object.get(name).?.object.get("samples").?.integer;
}

fn getCommonCategorySamples(result: Value, name: []const u8) i64 {
    return result.object.get("common_gating_universe").?.object
        .get("categories").?.object.get(name).?.object.get("samples").?.integer;
}

test "list add compare without trap proof remains unknown" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const result = try runTestAnalyze(
        allocator,
        &.{
            "add x17, x15, #0x4",
            "cmp x17, x16",
            "b.ls #0x14",
            "ldr w0, [x20, x15]",
            "ret",
            "mov x0, x17",
            "ret",
        },
        &.{ 0, 4 },
        &.{ .{ .offset = 0, .samples = 5129 }, .{ .offset = 4, .samples = 194 } },
        68829,
    );
    try std.testing.expectEqual(@as(i64, 5129), getCommonCategorySamples(result, "unknown") - 194);
    try std.testing.expectEqual(
        @as(i64, 0),
        result.object.get("cfg").?.object.get("structural_address_guard_branches").?.integer,
    );
}

test "list reference compare and xor are not assumed address" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const result = try runTestAnalyze(
        allocator,
        &.{
            "and w3, w23, #0xffff",
            "cmp w2, w3",
            "b.eq #0x14",
            "eor w2, w2, w3",
            "cmp w2, #0",
            "b.eq #0x1c",
            "ret",
            "ret",
        },
        &.{ 0, 4, 12, 16 },
        &.{
            .{ .offset = 0, .samples = 308 },
            .{ .offset = 4, .samples = 1318 },
            .{ .offset = 12, .samples = 894 },
            .{ .offset = 16, .samples = 83 },
        },
        33113,
    );
    try std.testing.expectEqual(@as(i64, 2603), getCommonCategorySamples(result, "unknown"));
    try std.testing.expectEqual(@as(i64, 0), getCommonCategorySamples(result, "address_generation"));
}

test "w and x aliases and extended index are address generation" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const result = try runTestAnalyze(
        allocator,
        &.{ "add w9, w0, #1", "ldr w10, [x2, w9, uxtw #2]", "ret" },
        &.{0},
        &.{.{ .offset = 0, .samples = 17 }},
        100,
    );
    try std.testing.expectEqual(@as(i64, 17), getCategorySamples(result, "address_generation"));
    try std.testing.expectEqual(@as(i64, 0), getCategorySamples(result, "unknown"));
}

test "dead flag output does not poison address chain" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const result = try runTestAnalyze(
        allocator,
        &.{ "add x9, x0, #1", "adds x10, x9, #4", "ldr w11, [x2, x10]", "ret" },
        &.{ 0, 4 },
        &.{ .{ .offset = 0, .samples = 6 }, .{ .offset = 4, .samples = 7 } },
        100,
    );
    try std.testing.expectEqual(@as(i64, 13), getCategorySamples(result, "address_generation"));
}

test "address value escapes and mixed control" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const stored = try runTestAnalyze(
        allocator,
        &.{ "add x9, x1, x2", "ldr w10, [x3, x9]", "str x9, [x4]", "ret" },
        &.{0},
        &.{.{ .offset = 0, .samples = 11 }},
        100,
    );
    try std.testing.expectEqual(@as(i64, 11), getCategorySamples(stored, "unknown"));

    const mixed = try runTestAnalyze(
        allocator,
        &.{ "add x9, x1, x2", "ldr w10, [x3, x9]", "cbnz x9, #0x10", "ret", "ret" },
        &.{0},
        &.{.{ .offset = 0, .samples = 11 }},
        100,
    );
    try std.testing.expectEqual(@as(i64, 11), getCategorySamples(mixed, "mixed"));

    const untyped_store = try runTestAnalyze(
        allocator,
        &.{ "add w9, w1, w2", "str w9, [x3]", "ret" },
        &.{0},
        &.{.{ .offset = 0, .samples = 13 }},
        100,
    );
    try std.testing.expectEqual(@as(i64, 13), getCategorySamples(untyped_store, "unknown"));

    const sig_ret = try runTestAnalyze(
        allocator,
        &.{ "add x0, x1, #8", "ldr w9, [x2, x0]", "ret" },
        &.{0},
        &.{.{ .offset = 0, .samples = 7 }},
        100,
    );
    try std.testing.expectEqual(@as(i64, 7), getCategorySamples(sig_ret, "unknown"));

    const spilled = try runTestAnalyze(
        allocator,
        &.{ "add x9, x1, #8", "str x9, [sp, #16]", "ldr x10, [sp, #16]", "ldr w0, [x10]", "ret" },
        &.{0},
        &.{.{ .offset = 0, .samples = 17 }},
        100,
    );
    try std.testing.expectEqual(@as(i64, 17), getCategorySamples(spilled, "unknown"));

    const direct_ret = try runTestAnalyze(
        allocator,
        &.{ "add x0, x1, #8", "ret" },
        &.{0},
        &.{.{ .offset = 0, .samples = 13 }},
        100,
    );
    try std.testing.expectEqual(@as(i64, 13), getCategorySamples(direct_ret, "unknown"));
}

test "structural address guard and invalidations" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    var diag: Diagnostic = .{};

    const guard = try runTestAnalyze(
        allocator,
        &.{ "cmp x9, x10", "b.hs #0x10", "ldr w0, [x1, x9, lsl #2]", "ret", "brk #0" },
        &.{0},
        &.{.{ .offset = 0, .samples = 19 }},
        100,
    );
    try std.testing.expectEqual(@as(i64, 19), getCategorySamples(guard, "structural_address_guard"));
    try std.testing.expectEqual(
        @as(i64, 1),
        guard.object.get("cfg").?.object.get("structural_address_guard_branches").?.integer,
    );
    const gate = try compareEngineAnalyses(allocator, guard, guard, DEFAULT_GATE_THRESHOLD_PCT, &diag);
    try std.testing.expect(gate.object.get("categories").?.object.get("structural_address_guard") == null);

    const call_trap = try runTestAnalyze(
        allocator,
        &.{ "cmp x9, x10", "b.hs #0x10", "ldr w0, [x1, x9]", "ret", "bl #0x1000", "ret" },
        &.{0},
        &.{.{ .offset = 0, .samples = 5 }},
        100,
    );
    try std.testing.expectEqual(@as(i64, 5), getCategorySamples(call_trap, "unknown"));

    const flag_clobber = try runTestAnalyze(
        allocator,
        &.{ "cmp x9, x10", "adds x11, x11, #1", "b.eq #0x10", "ret", "brk #0" },
        &.{ 0, 4 },
        &.{ .{ .offset = 0, .samples = 3 }, .{ .offset = 4, .samples = 4 } },
        100,
    );
    try std.testing.expectEqual(@as(i64, 7), getCategorySamples(flag_clobber, "unknown"));

    const float_cmp = try runTestAnalyze(
        allocator,
        &.{ "cmp x9, x10", "fcmp d0, d1", "b.hs #0x14", "ldr w0, [x1, x9]", "ret", "brk #0" },
        &.{0},
        &.{.{ .offset = 0, .samples = 23 }},
        100,
    );
    try std.testing.expectEqual(@as(i64, 23), getCategorySamples(float_cmp, "unknown"));

    const ccmp = try runTestAnalyze(
        allocator,
        &.{ "cmp x9, x10", "ccmp x11, x12, #0, eq", "b.hs #0x14", "ldr w0, [x1, x9]", "ret", "brk #0" },
        &.{0},
        &.{.{ .offset = 0, .samples = 21 }},
        100,
    );
    try std.testing.expectEqual(@as(i64, 21), getCategorySamples(ccmp, "unknown"));

    const lit_load = try runTestAnalyze(
        allocator,
        &.{ "cmp x9, x10", "b.hs #0x14", "ldr x9, 0x100", "ldr w0, [x1, x9]", "ret", "brk #0" },
        &.{0},
        &.{.{ .offset = 0, .samples = 31 }},
        100,
    );
    try std.testing.expectEqual(@as(i64, 31), getCategorySamples(lit_load, "unknown"));

    const mystery = try runTestAnalyze(
        allocator,
        &.{ "cmp x9, x10", "mystery d0, d1", "b.hs #0x14", "ldr w0, [x1, x9]", "ret", "brk #0" },
        &.{0},
        &.{.{ .offset = 0, .samples = 29 }},
        100,
    );
    try std.testing.expectEqual(@as(i64, 29), getCategorySamples(mystery, "unknown"));
}

test "cfg joins loops calls and writebacks" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const join = try runTestAnalyze(
        allocator,
        &.{ "cbz x0, #0xc", "add x9, x1, #4", "b #0x10", "sub x9, x1, #4", "ldr w2, [x3, x9]", "ret" },
        &.{ 4, 12 },
        &.{ .{ .offset = 4, .samples = 6 }, .{ .offset = 12, .samples = 8 } },
        100,
    );
    try std.testing.expectEqual(@as(i64, 14), getCategorySamples(join, "unknown"));

    const loop = try runTestAnalyze(
        allocator,
        &.{ "mov x9, xzr", "add x9, x9, #1", "ldr w10, [x2, x9]", "cmp x9, x3", "b.lo #0x4", "ret" },
        &.{ 4, 12 },
        &.{ .{ .offset = 4, .samples = 9 }, .{ .offset = 12, .samples = 2 } },
        100,
    );
    try std.testing.expectEqual(@as(i64, 11), getCategorySamples(loop, "unknown"));

    const caller_clobber = try runTestAnalyze(
        allocator,
        &.{ "add x9, x1, #4", "bl #0x1000", "ldr w0, [x2, x9]", "ret" },
        &.{0},
        &.{.{ .offset = 0, .samples = 4 }},
        100,
    );
    try std.testing.expectEqual(@as(i64, 4), getCategorySamples(caller_clobber, "unknown"));

    const callee_saved = try runTestAnalyze(
        allocator,
        &.{ "add x19, x1, #4", "bl #0x1000", "ldr w0, [x2, x19]", "ret" },
        &.{0},
        &.{.{ .offset = 0, .samples = 4 }},
        100,
    );
    try std.testing.expectEqual(@as(i64, 4), getCategorySamples(callee_saved, "address_generation"));

    const post_idx = try runTestAnalyze(
        allocator,
        &.{ "add x9, x1, #4", "ldr x0, [x9], #8", "str x9, [x2]", "ret" },
        &.{0},
        &.{.{ .offset = 0, .samples = 10 }},
        100,
    );
    try std.testing.expectEqual(@as(i64, 10), getCategorySamples(post_idx, "unknown"));
}

test "common universe and gate comparisons" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    var diag: Diagnostic = .{};

    const wamr_insts = [_]InstructionInput{
        try makeTestInst(allocator, 0, "madd x9, x1, x2, x3"),
        try makeTestInst(allocator, 4, "ldr w0, [x4, x9]"),
        try makeTestInst(allocator, 8, "ret"),
    };
    const wt_insts = [_]InstructionInput{
        try makeTestInst(allocator, 0, "umaddl x9, w1, w2, x3"),
        try makeTestInst(allocator, 4, "ldr w0, [x4, x9]"),
        try makeTestInst(allocator, 8, "ret"),
    };
    const wamr = try analyzeInstructionStream(allocator, &wamr_insts, .{
        .broad_classes = &.{ "alu", "other", "other" },
        .samples_by_offset = &.{.{ .offset = 0, .samples = 6 }},
        .total_run_samples = 100,
    }, &diag);
    const wasmtime = try analyzeInstructionStream(allocator, &wt_insts, .{
        .broad_classes = &.{ "other", "other", "other" },
        .samples_by_offset = &.{.{ .offset = 0, .samples = 6 }},
        .total_run_samples = 100,
    }, &diag);
    try std.testing.expectEqual(@as(i64, 6), wamr.object.get("broad_alu_samples").?.integer);
    try std.testing.expectEqual(@as(i64, 0), wasmtime.object.get("broad_alu_samples").?.integer);
    try std.testing.expectEqual(@as(i64, 6), getCommonCategorySamples(wamr, "address_generation"));
    try std.testing.expectEqual(@as(i64, 6), getCommonCategorySamples(wasmtime, "address_generation"));

    const gate = try compareEngineAnalyses(allocator, wamr, wasmtime, DEFAULT_GATE_THRESHOLD_PCT, &diag);
    try std.testing.expect(!gate.object.get("optimization_authorized").?.bool);

    const wamr_clear = try runTestAnalyze(
        allocator,
        &.{ "add x9, x1, #4", "ldr w0, [x2, x9]", "ret" },
        &.{0},
        &.{.{ .offset = 0, .samples = 8 }},
        100,
    );
    const wt_clear = try runTestAnalyze(
        allocator,
        &.{ "add x9, x1, #4", "ldr w0, [x2, x9]", "ret" },
        &.{0},
        &.{.{ .offset = 0, .samples = 1 }},
        100,
    );
    const gate_clear = try compareEngineAnalyses(allocator, wamr_clear, wt_clear, DEFAULT_GATE_THRESHOLD_PCT, &diag);
    try std.testing.expect(gate_clear.object.get("optimization_authorized").?.bool);
}

test "classify unknown reason clusters and terminal sink exhaustion" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    var diag: Diagnostic = .{};

    const cluster_res = try runTestAnalyze(
        allocator,
        &.{
            "add x9, x1, #4",
            "ldr w2, [x3, x9]",
            "cmp w4, w5",
            "b.eq #0x1c",
            "mov x10, x1",
            "b #0x20",
            "nop",
            "mov x10, x2",
            "str w10, [x3]",
            "mov x11, x4",
            "mov x0, x5",
            "ret",
        },
        &.{ 0, 8, 16, 28, 36, 40 },
        &.{
            .{ .offset = 0, .samples = 10 },
            .{ .offset = 8, .samples = 20 },
            .{ .offset = 16, .samples = 15 },
            .{ .offset = 36, .samples = 5 },
            .{ .offset = 40, .samples = 7 },
        },
        100,
    );
    const clusters = try classifyUnknownReasonClusters(allocator, cluster_res, "analysis", &diag);
    try std.testing.expectEqual(@as(i64, 47), clusters.object.get("uncertain_samples").?.integer);
    const cl_map = clusters.object.get("clusters").?.object;
    try std.testing.expectEqual(@as(i64, 20), cl_map.get("non_trap_control").?.object.get("samples").?.integer);
    try std.testing.expectEqual(@as(i64, 15), cl_map.get("ambiguous_cfg_join").?.object.get("samples").?.integer);
    try std.testing.expectEqual(@as(i64, 7), cl_map.get("untyped_escape").?.object.get("samples").?.integer);
    try std.testing.expectEqual(@as(i64, 5), cl_map.get("dead_or_unconsumed").?.object.get("samples").?.integer);

    const wamr_insts = [_]InstructionInput{
        try makeTestInst(allocator, 0, "add x9, x1, #4"),
        try makeTestInst(allocator, 4, "ldr w0, [x2, x9]"),
        try makeTestInst(allocator, 8, "ret"),
    };
    const wamr_ex = try analyzeTerminalSinkExhaustion(allocator, &wamr_insts, .{
        .broad_classes = &.{ "alu", "linear_memory", "other" },
        .samples_by_offset = &.{.{ .offset = 0, .samples = 10 }},
        .total_run_samples = 100,
    }, &diag);

    const wt_insts = [_]InstructionInput{
        try makeTestInst(allocator, 0, "cbz w0, #0xc"),
        try makeTestInst(allocator, 4, "add x9, x1, #4"),
        try makeTestInst(allocator, 8, "b #0x10"),
        try makeTestInst(allocator, 12, "add x9, x1, #8"),
        try makeTestInst(allocator, 16, "ldr w2, [x3, x9]"),
        try makeTestInst(allocator, 20, "cmp w2, #0"),
        try makeTestInst(allocator, 24, "b.eq #0x20"),
        try makeTestInst(allocator, 28, "ret"),
        try makeTestInst(allocator, 32, "ret"),
    };
    const wt_ex = try analyzeTerminalSinkExhaustion(allocator, &wt_insts, .{
        .broad_classes = &.{ "branch", "alu", "branch", "alu", "linear_memory", "alu", "branch", "other", "other" },
        .samples_by_offset = &.{
            .{ .offset = 4, .samples = 2 },
            .{ .offset = 12, .samples = 3 },
            .{ .offset = 20, .samples = 15 },
        },
        .total_run_samples = 100,
    }, &diag);

    const comparison = try compareTerminalSinkExhaustion(
        allocator,
        wamr_ex.object.get("primary_analysis").?,
        wt_ex.object.get("primary_analysis").?,
        wt_ex,
        wamr_ex,
        DEFAULT_GATE_THRESHOLD_PCT,
        &diag,
    );
    const addr = comparison.object.get("categories").?.object.get("address_generation").?.object;
    try std.testing.expectEqual(@as(i64, 0), addr.get("strict_wasmtime_samples").?.integer);
    try std.testing.expectEqual(@as(i64, 5), addr.get("unblocked_wasmtime_samples").?.integer);
    try std.testing.expectEqual(@as(i64, 5), addr.get("wasmtime_reclassified_from_unknown_samples").?.integer);
    try std.testing.expectEqual(@as(i64, 15), addr.get("wasmtime_remaining_uncertain_common_samples").?.integer);
    try std.testing.expect(comparison.object.get("proven_exhausted").?.bool);
    try std.testing.expect(!comparison.object.get("optimization_authorized").?.bool);
}

test "move aliases pointer auth external targets and validation edge cases" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    var diag: Diagnostic = .{};

    // Move aliases cannot disappear from reference universe
    const wamr_mov_insts = [_]InstructionInput{
        try makeTestInst(allocator, 0, "add x9, x1, #0"),
        try makeTestInst(allocator, 4, "ldr w0, [x2, x9]"),
        try makeTestInst(allocator, 8, "ret"),
    };
    const wt_mov_insts = [_]InstructionInput{
        try makeTestInst(allocator, 0, "mov x9, x1"),
        try makeTestInst(allocator, 4, "ldr w0, [x2, x9]"),
        try makeTestInst(allocator, 8, "ret"),
    };
    const wamr_mov = try analyzeInstructionStream(allocator, &wamr_mov_insts, .{
        .broad_classes = &.{ "alu", "other", "other" },
        .samples_by_offset = &.{.{ .offset = 0, .samples = 6 }},
        .total_run_samples = 100,
    }, &diag);
    const wt_mov = try analyzeInstructionStream(allocator, &wt_mov_insts, .{
        .broad_classes = &.{ "move_ext", "other", "other" },
        .samples_by_offset = &.{.{ .offset = 0, .samples = 6 }},
        .total_run_samples = 100,
    }, &diag);
    try std.testing.expectEqual(@as(i64, 0), wt_mov.object.get("broad_alu_samples").?.integer);
    try std.testing.expectEqual(@as(i64, 6), getCommonCategorySamples(wt_mov, "address_generation"));
    const mov_gate = try compareEngineAnalyses(allocator, wamr_mov, wt_mov, DEFAULT_GATE_THRESHOLD_PCT, &diag);
    try std.testing.expect(!mov_gate.object.get("optimization_authorized").?.bool);

    // Pointer auth and hint aliases invalidate guards
    for ([_][]const u8{
        "pacia x9, sp",
        "autia x9, x10",
        "pacdza x9",
        "autdzb x9",
        "xpaci x9",
        "xpaclri",
        "autia1716",
        "pacibsp",
        "hint #0x7",
    }) |pauth| {
        const reg = if (std.mem.endsWith(u8, pauth, "1716"))
            "x17"
        else if (std.mem.endsWith(u8, pauth, "lri") or std.mem.endsWith(u8, pauth, "bsp") or std.mem.startsWith(u8, pauth, "hint"))
            "x30"
        else
            "x9";
        const cmp_inst = try std.fmt.allocPrint(allocator, "cmp {s}, x10", .{reg});
        const ldr_inst = try std.fmt.allocPrint(allocator, "ldr w0, [x1, {s}]", .{reg});
        const p_res = try runTestAnalyze(
            allocator,
            &.{ cmp_inst, "b.hs #0x14", pauth, ldr_inst, "ret", "brk #0" },
            &.{0},
            &.{.{ .offset = 0, .samples = 17 }},
            100,
        );
        try std.testing.expectEqual(@as(i64, 0), getCategorySamples(p_res, "structural_address_guard"));
        try std.testing.expectEqual(@as(i64, 17), getCategorySamples(p_res, "unknown"));
    }

    // Absolute external target cannot alias a relative trap offset
    const abs_insts = [_]InstructionInput{
        .{ .offset = 0, .address = 0x1000, .size = 4, .mnemonic = "cmp", .operands = "x9, x10", .text = "cmp x9, x10" },
        .{ .offset = 4, .address = 0x1004, .size = 4, .mnemonic = "b.hs", .operands = "#0x10", .text = "b.hs #0x10" },
        .{ .offset = 8, .address = 0x1008, .size = 4, .mnemonic = "ldr", .operands = "w0, [x1, x9]", .text = "ldr w0, [x1, x9]" },
        .{ .offset = 12, .address = 0x100c, .size = 4, .mnemonic = "ret", .operands = "", .text = "ret" },
        .{ .offset = 16, .address = 0x1010, .size = 4, .mnemonic = "brk", .operands = "#0", .text = "brk #0" },
    };
    const abs_res = try analyzeInstructionStream(allocator, &abs_insts, .{
        .broad_classes = &.{ "alu", "other", "other", "other", "other" },
        .samples_by_offset = &.{.{ .offset = 0, .samples = 17 }},
        .total_run_samples = 100,
        .address_base = 0x1000,
    }, &diag);
    try std.testing.expectEqual(@as(i64, 0), getCategorySamples(abs_res, "structural_address_guard"));
    try std.testing.expectEqual(@as(i64, 17), getCategorySamples(abs_res, "unknown"));
    try std.testing.expectEqual(
        @as(usize, 1),
        abs_res.object.get("cfg").?.object.get("external_branch_targets").?.array.items.len,
    );

    // Sample counts must fit the global mapping
    const fit_insts = [_]InstructionInput{
        try makeTestInst(allocator, 0, "add x9, x1, #4"),
        try makeTestInst(allocator, 4, "ldr w0, [x2, x9]"),
        try makeTestInst(allocator, 8, "ret"),
    };
    try std.testing.expectError(error.ProvenanceFailed, analyzeInstructionStream(allocator, &fit_insts, .{
        .broad_classes = &.{ "alu", "other", "other" },
        .samples_by_offset = &.{.{ .offset = 0, .samples = 10 }},
        .total_run_samples = 100,
        .global_attributed_samples = 9,
    }, &diag));
    try std.testing.expectError(error.ProvenanceFailed, analyzeInstructionStream(allocator, &fit_insts, .{
        .broad_classes = &.{ "alu", "other", "other" },
        .samples_by_offset = &.{.{ .offset = 0, .samples = 10 }},
        .total_run_samples = 100,
        .global_attributed_samples = 101,
    }, &diag));

    // Opaque other opcode is common universe uncertainty
    const opq_insts = [_]InstructionInput{
        try makeTestInst(allocator, 0, "mystery x9, x1, x2"),
        try makeTestInst(allocator, 4, "ldr w0, [x3, x9]"),
        try makeTestInst(allocator, 8, "ret"),
    };
    const opq_res = try analyzeInstructionStream(allocator, &opq_insts, .{
        .broad_classes = &.{ "other", "other", "other" },
        .samples_by_offset = &.{.{ .offset = 0, .samples = 6 }},
        .total_run_samples = 100,
    }, &diag);
    try std.testing.expectEqual(@as(i64, 0), opq_res.object.get("broad_alu_samples").?.integer);
    try std.testing.expectEqual(
        @as(i64, 6),
        opq_res.object.get("common_gating_universe").?.object.get("samples").?.integer,
    );
    try std.testing.expectEqual(@as(i64, 6), getCommonCategorySamples(opq_res, "unknown"));

    // Implicit transforms cannot escape reference uncertainty
    for ([_][]const u8{
        "adcs x9, x1, x2",
        "sbcs x9, x1, x2",
        "ngc x9, x1",
        "cinc x9, x1, eq",
        "csel x9, x1, x2, eq",
        "ccmp x9, x10, #0, eq",
        "adr x9, #0x10",
        "adrp x9, #0x1000",
        "crc32w w9, w1, w2",
    }) |inst_txt| {
        const imp_insts = [_]InstructionInput{
            try makeTestInst(allocator, 0, inst_txt),
            try makeTestInst(allocator, 4, "ret"),
        };
        const imp_res = try analyzeInstructionStream(allocator, &imp_insts, .{
            .broad_classes = &.{ "other", "other" },
            .samples_by_offset = &.{.{ .offset = 0, .samples = 6 }},
            .total_run_samples = 100,
        }, &diag);
        try std.testing.expectEqual(
            @as(i64, 6),
            imp_res.object.get("common_gating_universe").?.object.get("samples").?.integer,
        );
        try std.testing.expectEqual(@as(i64, 6), getCommonCategorySamples(imp_res, "unknown"));
    }

    // Unavailable engine blocks gate
    const unavail = try unavailableAnalysis(allocator, "missing hot function", 100);
    const gate_unavail = try compareEngineAnalyses(allocator, wamr_mov, unavail, DEFAULT_GATE_THRESHOLD_PCT, &diag);
    try std.testing.expectEqualStrings("blocked", gate_unavail.object.get("status").?.string);
    try std.testing.expect(!gate_unavail.object.get("optimization_authorized").?.bool);

    // Malformed disassembly fails closed
    try std.testing.expectError(error.ProvenanceFailed, analyzeInstructionStream(allocator, &.{}, .{
        .broad_classes = &.{},
        .samples_by_offset = &.{},
        .total_run_samples = 100,
    }, &diag));
}
