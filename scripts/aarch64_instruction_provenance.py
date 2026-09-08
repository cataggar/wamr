#!/usr/bin/env python3
"""Conservative AArch64 ALU provenance from native CFG and def-use paths.

The analysis deliberately uses only architectural instruction semantics.  It
does not assign meaning to engine-selected registers, nearby instructions, or
trap-looking branch targets.  Every ALU instruction is placed in exactly one
of address generation, structural address guard, algorithmic ALU, mixed, or
unknown.  A separate complete common gating universe prevents legacy display
classifier differences from becoming optimizer evidence.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 2
ANALYSIS_KIND = "aarch64-conservative-alu-provenance"
CATEGORIES = (
    "address_generation",
    "structural_address_guard",
    "algorithmic_alu",
    "mixed",
    "unknown",
)
GATE_CATEGORIES = (
    "address_generation",
    "algorithmic_alu",
)
PROVEN_CATEGORIES = (
    "address_generation",
    "structural_address_guard",
    "algorithmic_alu",
)
DEFAULT_GATE_THRESHOLD_PCT = 5.0
CALLER_CLOBBERED = tuple(f"x{index}" for index in range(19)) + ("x30", "nzcv")
POSSIBLE_CALL_ARGUMENTS = tuple(f"x{index}" for index in range(8))
POSSIBLE_RETURN_VALUES = ("x0", "x1")
TRACKED_REGISTERS = (
    tuple(f"x{index}" for index in range(31)) + ("sp", "nzcv")
)

REGISTER_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(w(?:[12]?\d|30)|x(?:[12]?\d|30)|wsp|sp|wzr|xzr|fp|lr)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
INTEGER_RE = re.compile(r"[-+]?(?:0x[0-9a-f]+|\d+)", re.IGNORECASE)

LOAD_PREFIXES = (
    "ldr",
    "ldur",
    "ldp",
    "ldnp",
    "ldtr",
    "ldxr",
    "ldaxr",
    "ldar",
    "ldapr",
)
STORE_PREFIXES = (
    "str",
    "stur",
    "stp",
    "stnp",
    "sttr",
    "stxr",
    "stlxr",
    "stlr",
)
ALU_DEST_MNEMONICS = {
    "add",
    "adds",
    "sub",
    "subs",
    "adc",
    "adcs",
    "sbc",
    "sbcs",
    "and",
    "ands",
    "orr",
    "orn",
    "eor",
    "eon",
    "bic",
    "bics",
    "madd",
    "msub",
    "mul",
    "mneg",
    "smaddl",
    "smsubl",
    "umaddl",
    "umsubl",
    "smulh",
    "umulh",
    "smull",
    "umull",
    "udiv",
    "sdiv",
    "lsl",
    "lsr",
    "asr",
    "ror",
    "extr",
    "ubfm",
    "sbfm",
    "bfm",
    "ubfx",
    "sbfx",
    "bfi",
    "bfxil",
    "uxtb",
    "uxth",
    "uxtw",
    "sxtb",
    "sxth",
    "sxtw",
    "csel",
    "csinc",
    "csinv",
    "csneg",
    "cset",
    "csetm",
    "neg",
    "negs",
    "mvn",
    "adr",
    "adrp",
}
FLAG_ONLY_MNEMONICS = {"cmp", "cmn", "tst"}
CONDITIONAL_FLAG_MNEMONICS = {"ccmp", "ccmn"}
FLOAT_FLAG_MNEMONICS = {"fcmp", "fcmpe", "fccmp", "fccmpe"}
FLAG_WRITING_MNEMONICS = {
    "adds",
    "subs",
    "adcs",
    "sbcs",
    "ands",
    "bics",
    "negs",
}
MOVE_MNEMONICS = {
    "mov",
    "movz",
    "movn",
    "movk",
    "fmov",
}
VALUE_TRANSFORM_MNEMONICS = {
    "rev",
    "rev16",
    "rev32",
    "rbit",
    "clz",
    "cls",
}
TRAP_MNEMONICS = {"brk", "hlt", "udf"}
SIDE_EFFECT_FREE_TRAP_PATH_MNEMONICS = {"nop", "hint", "paciasp", "autiasp"}


class ProvenanceError(RuntimeError):
    pass


@dataclass(frozen=True)
class Instruction:
    offset: int
    size: int
    mnemonic: str
    operands: str
    text: str
    address: int


@dataclass(frozen=True)
class Use:
    register: str
    role: str
    propagate_to: tuple[str, ...] = ()


@dataclass
class Effect:
    uses: list[Use] = field(default_factory=list)
    definitions: set[str] = field(default_factory=set)
    primary_definitions: set[str] = field(default_factory=set)
    secondary_definitions: set[str] = field(default_factory=set)
    branch_kind: str | None = None
    branch_target: int | None = None
    trap: bool = False
    memory: bool = False
    recognized: bool = True
    gating_candidate: bool = False
    opaque_barrier: bool = False


@dataclass(frozen=True)
class Definition:
    instruction: int
    register: str


@dataclass(frozen=True)
class Consumer:
    instruction: int
    use: Use
    ambiguous: bool


@dataclass
class Trace:
    roles: set[str] = field(default_factory=set)
    unknown_reasons: set[str] = field(default_factory=set)
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def merge(self, other: "Trace") -> None:
        self.roles.update(other.roles)
        self.unknown_reasons.update(other.unknown_reasons)
        self.evidence.extend(other.evidence)


def split_operands(text: str) -> list[str]:
    result: list[str] = []
    current: list[str] = []
    depth = 0
    for char in text:
        if char in "[(":
            depth += 1
        elif char in "])":
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            result.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current or text.strip():
        result.append("".join(current).strip())
    return [item for item in result if item]


def split_instruction_text(text: str) -> tuple[str, str]:
    parts = text.strip().split(None, 1)
    if not parts:
        return "", ""
    return parts[0].lower(), parts[1].strip() if len(parts) == 2 else ""


def canonical_register(value: str) -> str | None:
    register = value.lower()
    if register in {"xzr", "wzr"}:
        return None
    if register == "fp":
        return "x29"
    if register == "lr":
        return "x30"
    if register in {"sp", "wsp"}:
        return "sp"
    if register.startswith(("w", "x")) and register[1:].isdigit():
        index = int(register[1:])
        if 0 <= index <= 30:
            return f"x{index}"
    return None


def registers_in_operand(operand: str) -> list[str]:
    result: list[str] = []
    for match in REGISTER_RE.finditer(operand):
        register = canonical_register(match.group(1))
        if register is not None and register not in result:
            result.append(register)
    return result


def normalize_instructions(
    instructions: Sequence[Any], *, address_base: int = 0
) -> list[Instruction]:
    normalized: list[Instruction] = []
    previous = -1
    for item in instructions:
        try:
            offset = int(item.offset)
            size = int(item.size)
            text = str(item.text).strip()
        except (AttributeError, TypeError, ValueError) as exc:
            raise ProvenanceError("malformed disassembly instruction") from exc
        if offset <= previous or size <= 0 or not text:
            raise ProvenanceError(
                "disassembly instructions must have increasing offsets, "
                "positive sizes, and nonempty text"
            )
        mnemonic = str(getattr(item, "mnemonic", "")).lower()
        operands = str(getattr(item, "operands", ""))
        if not mnemonic:
            mnemonic, operands = split_instruction_text(text)
        address = int(getattr(item, "address", address_base + offset))
        normalized.append(
            Instruction(
                offset=offset,
                size=size,
                mnemonic=mnemonic,
                operands=operands,
                text=text,
                address=address,
            )
        )
        previous = offset
    if not normalized:
        raise ProvenanceError("disassembly contained no instructions")
    return normalized


def _add_uses(
    effect: Effect,
    operands: Iterable[str],
    role: str,
    propagate_to: tuple[str, ...] = (),
) -> None:
    for operand in operands:
        for register in registers_in_operand(operand):
            effect.uses.append(Use(register, role, propagate_to))


def _parse_target(operand: str) -> int | None:
    match = INTEGER_RE.search(operand.replace("#", ""))
    if match is None:
        return None
    try:
        return int(match.group(0), 0)
    except ValueError:
        return None


def _is_load(mnemonic: str) -> bool:
    return mnemonic.startswith(LOAD_PREFIXES)


def _is_store(mnemonic: str) -> bool:
    return mnemonic.startswith(STORE_PREFIXES)


def _opaque_effect(*, gating_candidate: bool) -> Effect:
    effect = Effect(
        recognized=False,
        gating_candidate=gating_candidate,
        opaque_barrier=True,
        branch_kind="opaque",
    )
    for register in TRACKED_REGISTERS:
        effect.uses.append(Use(register, "unknown"))
    effect.definitions.update(TRACKED_REGISTERS)
    effect.secondary_definitions.update(TRACKED_REGISTERS)
    return effect


def _memory_effect(instruction: Instruction, operands: list[str]) -> Effect:
    effect = Effect(memory=True)
    memory_indices = [
        index for index, operand in enumerate(operands) if "[" in operand
    ]
    if (
        not memory_indices
        and _is_load(instruction.mnemonic)
        and len(operands) >= 2
    ):
        destination = registers_in_operand(operands[0])
        if len(destination) == 1:
            effect.definitions.add(destination[0])
            effect.primary_definitions.add(destination[0])
        return effect
    if len(memory_indices) != 1:
        return _opaque_effect(gating_candidate=False)
    memory_index = memory_indices[0]
    memory_operand = operands[memory_index]
    address_registers = registers_in_operand(memory_operand)
    for register in address_registers:
        effect.uses.append(Use(register, "address"))

    load = _is_load(instruction.mnemonic)
    store = _is_store(instruction.mnemonic)
    if not load and not store:
        return _opaque_effect(gating_candidate=False)

    value_operands = operands[:memory_index]
    if load:
        for operand in value_operands:
            registers = registers_in_operand(operand)
            if len(registers) == 1:
                effect.definitions.add(registers[0])
                effect.primary_definitions.add(registers[0])
    else:
        status_result = instruction.mnemonic.startswith(("stxr", "stlxr"))
        if status_result and value_operands:
            registers = registers_in_operand(value_operands[0])
            if len(registers) == 1:
                effect.definitions.add(registers[0])
                effect.primary_definitions.add(registers[0])
            else:
                effect.recognized = False
            value_operands = value_operands[1:]
        _add_uses(effect, value_operands, "escape")

    writeback = memory_operand.rstrip().endswith("]!")
    post_index_operands = operands[memory_index + 1 :]
    if post_index_operands:
        writeback = True
        _add_uses(effect, post_index_operands, "value")
    if writeback:
        if not address_registers:
            return _opaque_effect(gating_candidate=False)
        else:
            base = address_registers[0]
            effect.definitions.add(base)
            effect.secondary_definitions.add(base)
            effect.uses = [
                Use(
                    use.register,
                    use.role,
                    (base,) if use.register == base and use.role == "address" else (),
                )
                for use in effect.uses
            ]
    return effect


def instruction_effect(instruction: Instruction) -> Effect:
    mnemonic = instruction.mnemonic
    operands = split_operands(instruction.operands)

    if _is_load(mnemonic) or _is_store(mnemonic):
        return _memory_effect(instruction, operands)
    if mnemonic == "prfm":
        memory_operands = [
            operand for operand in operands if "[" in operand
        ]
        if len(memory_operands) != 1:
            return _opaque_effect(gating_candidate=False)
        effect = Effect(memory=True)
        _add_uses(effect, memory_operands, "address")
        return effect

    effect = Effect()
    if mnemonic in TRAP_MNEMONICS:
        effect.trap = True
        effect.branch_kind = "trap"
        return effect
    if mnemonic in SIDE_EFFECT_FREE_TRAP_PATH_MNEMONICS:
        return effect
    if mnemonic == "b":
        effect.branch_kind = "unconditional"
        effect.branch_target = _parse_target(operands[-1]) if operands else None
        return effect
    if mnemonic.startswith("b."):
        effect.branch_kind = "conditional"
        effect.branch_target = _parse_target(operands[-1]) if operands else None
        effect.uses.append(Use("nzcv", "control"))
        return effect
    if mnemonic in {"cbz", "cbnz", "tbz", "tbnz"}:
        effect.branch_kind = "conditional"
        effect.branch_target = _parse_target(operands[-1]) if operands else None
        if operands:
            _add_uses(effect, operands[:1], "control")
        return effect
    if mnemonic == "br":
        effect.branch_kind = "indirect"
        _add_uses(effect, operands[:1], "control")
        for register in TRACKED_REGISTERS:
            if register not in {"sp", "nzcv"}:
                effect.uses.append(Use(register, "unknown"))
        return effect
    if mnemonic in {"bl", "blr"}:
        effect.branch_kind = "call"
        if mnemonic == "blr":
            _add_uses(effect, operands[:1], "control")
        for register in POSSIBLE_CALL_ARGUMENTS:
            effect.uses.append(Use(register, "unknown"))
        effect.definitions.update(CALLER_CLOBBERED)
        effect.secondary_definitions.update(CALLER_CLOBBERED)
        return effect
    if mnemonic == "ret":
        effect.branch_kind = "return"
        if operands:
            _add_uses(effect, operands[:1], "control")
        else:
            effect.uses.append(Use("x30", "control"))
        for register in POSSIBLE_RETURN_VALUES:
            effect.uses.append(Use(register, "escape"))
        return effect

    if mnemonic in CONDITIONAL_FLAG_MNEMONICS:
        effect.gating_candidate = True
        effect.definitions.add("nzcv")
        effect.primary_definitions.add("nzcv")
        effect.uses.append(Use("nzcv", "unknown"))
        _add_uses(effect, operands, "unknown")
        return effect

    if mnemonic in FLAG_ONLY_MNEMONICS:
        effect.gating_candidate = True
        effect.definitions.add("nzcv")
        effect.primary_definitions.add("nzcv")
        _add_uses(effect, operands, "compare")
        return effect

    if mnemonic in FLOAT_FLAG_MNEMONICS:
        effect.definitions.add("nzcv")
        effect.primary_definitions.add("nzcv")
        if mnemonic.startswith("fcc"):
            effect.uses.append(Use("nzcv", "unknown"))
        return effect

    if mnemonic in ALU_DEST_MNEMONICS or mnemonic in VALUE_TRANSFORM_MNEMONICS:
        effect.gating_candidate = True
        if not operands:
            return _opaque_effect(gating_candidate=True)
        destination = registers_in_operand(operands[0])
        if len(destination) != 1:
            return _opaque_effect(gating_candidate=True)
        dest = destination[0]
        effect.definitions.add(dest)
        effect.primary_definitions.add(dest)
        propagated_definitions = [dest]
        if mnemonic in FLAG_WRITING_MNEMONICS:
            effect.definitions.add("nzcv")
            effect.secondary_definitions.add("nzcv")
            propagated_definitions.append("nzcv")
        sources = operands[1:]
        if mnemonic in {"cset", "csetm"}:
            effect.uses.append(Use("nzcv", "value", (dest,)))
        else:
            _add_uses(
                effect,
                sources,
                "value",
                tuple(propagated_definitions),
            )
            if mnemonic.startswith("cs"):
                effect.uses.append(Use("nzcv", "value", (dest,)))
        return effect

    if mnemonic in MOVE_MNEMONICS:
        if not operands:
            return _opaque_effect(gating_candidate=False)
        destination = registers_in_operand(operands[0])
        if len(destination) != 1:
            return _opaque_effect(gating_candidate=False)
        dest = destination[0]
        effect.definitions.add(dest)
        effect.primary_definitions.add(dest)
        sources = operands[1:]
        if mnemonic == "movk":
            effect.uses.append(Use(dest, "value", (dest,)))
        _add_uses(effect, sources, "value", (dest,))
        return effect

    if mnemonic == "mrs":
        if operands:
            destination = registers_in_operand(operands[0])
            if len(destination) == 1:
                effect.definitions.add(destination[0])
                effect.primary_definitions.add(destination[0])
        return effect
    if mnemonic == "msr":
        _add_uses(effect, operands, "unknown")
        if operands and operands[0].strip().lower() == "nzcv":
            effect.definitions.add("nzcv")
            effect.primary_definitions.add("nzcv")
        return effect

    return _opaque_effect(gating_candidate=True)


def _target_index(
    instructions: Sequence[Instruction],
    target: int | None,
) -> int | None:
    if target is None:
        return None
    by_address = {instruction.address: index for index, instruction in enumerate(instructions)}
    by_offset = {instruction.offset: index for index, instruction in enumerate(instructions)}
    return by_address.get(target, by_offset.get(target))


def build_cfg(
    instructions: Sequence[Instruction],
    effects: Sequence[Effect],
) -> tuple[list[set[int]], list[set[int]], list[dict[str, Any]]]:
    successors = [set() for _ in instructions]
    external_targets: list[dict[str, Any]] = []
    for index, effect in enumerate(effects):
        following = index + 1 if index + 1 < len(instructions) else None
        target_index = _target_index(instructions, effect.branch_target)
        if effect.branch_target is not None and target_index is None:
            external_targets.append(
                {
                    "offset": instructions[index].offset,
                    "instruction": instructions[index].text,
                    "target": effect.branch_target,
                }
            )
            if effect.branch_kind == "unconditional":
                for register in POSSIBLE_CALL_ARGUMENTS:
                    effect.uses.append(Use(register, "unknown"))
        if effect.branch_kind == "unconditional":
            if target_index is not None:
                successors[index].add(target_index)
        elif effect.branch_kind == "conditional":
            if following is not None:
                successors[index].add(following)
            if target_index is not None:
                successors[index].add(target_index)
        elif effect.branch_kind in {"indirect", "return", "trap", "opaque"}:
            pass
        elif following is not None:
            successors[index].add(following)
    predecessors = [set() for _ in instructions]
    for index, targets in enumerate(successors):
        for target in targets:
            predecessors[target].add(index)
    return successors, predecessors, external_targets


def _reachable(successors: Sequence[set[int]]) -> set[int]:
    result: set[int] = set()
    stack = [0] if successors else []
    while stack:
        current = stack.pop()
        if current in result:
            continue
        result.add(current)
        stack.extend(successors[current] - result)
    return result


def _reaching_definitions(
    effects: Sequence[Effect],
    predecessors: Sequence[set[int]],
    reachable: set[int],
) -> tuple[list[dict[str, frozenset[Definition]]], list[dict[str, frozenset[Definition]]]]:
    entry = {
        register: frozenset({Definition(-1, register)})
        for register in TRACKED_REGISTERS
    }
    incoming: list[dict[str, frozenset[Definition]]] = [
        {} for _ in effects
    ]
    outgoing: list[dict[str, frozenset[Definition]]] = [
        {} for _ in effects
    ]
    changed = True
    while changed:
        changed = False
        for index, effect in enumerate(effects):
            if index not in reachable:
                continue
            merged: dict[str, set[Definition]] = defaultdict(set)
            if index == 0:
                for register, definitions in entry.items():
                    merged[register].update(definitions)
            for predecessor in predecessors[index]:
                for register, definitions in outgoing[predecessor].items():
                    merged[register].update(definitions)
            next_incoming = {
                register: frozenset(definitions)
                for register, definitions in merged.items()
            }
            next_outgoing = dict(next_incoming)
            for register in effect.definitions:
                next_outgoing[register] = frozenset(
                    {Definition(index, register)}
                )
            if (
                next_incoming != incoming[index]
                or next_outgoing != outgoing[index]
            ):
                incoming[index] = next_incoming
                outgoing[index] = next_outgoing
                changed = True
    return incoming, outgoing


def _dominators(
    predecessors: Sequence[set[int]], reachable: set[int]
) -> list[set[int]]:
    all_reachable = set(reachable)
    dominators = [
        ({index} if index not in reachable else set(all_reachable))
        for index in range(len(predecessors))
    ]
    if not predecessors:
        return dominators
    dominators[0] = {0}
    changed = True
    while changed:
        changed = False
        for index in sorted(reachable - {0}):
            active_predecessors = predecessors[index] & reachable
            if not active_predecessors:
                next_value = {index}
            else:
                iterator = iter(active_predecessors)
                next_value = set(dominators[next(iterator)])
                for predecessor in iterator:
                    next_value.intersection_update(dominators[predecessor])
                next_value.add(index)
            if next_value != dominators[index]:
                dominators[index] = next_value
                changed = True
    return dominators


def _trap_only_path(
    start: int,
    effects: Sequence[Effect],
    successors: Sequence[set[int]],
    visiting: set[int],
    memo: dict[int, bool],
) -> bool:
    if start in memo:
        return memo[start]
    if start in visiting:
        return False
    effect = effects[start]
    if effect.trap:
        memo[start] = True
        return True
    if (
        not effect.recognized
        or effect.memory
        or effect.uses
        or effect.definitions
        or effect.branch_kind in {"call", "conditional", "indirect", "return"}
    ):
        memo[start] = False
        return False
    targets = successors[start]
    if not targets:
        memo[start] = False
        return False
    next_visiting = set(visiting)
    next_visiting.add(start)
    result = all(
        _trap_only_path(
            target, effects, successors, next_visiting, memo
        )
        for target in targets
    )
    memo[start] = result
    return result


def _use_sources(
    incoming: Sequence[dict[str, frozenset[Definition]]],
    instruction: int,
    use: Use,
) -> frozenset[Definition]:
    return incoming[instruction].get(use.register, frozenset())


def _structural_address_guard_branches(
    effects: Sequence[Effect],
    successors: Sequence[set[int]],
    incoming: Sequence[dict[str, frozenset[Definition]]],
    dominators: Sequence[set[int]],
    reachable: set[int],
) -> tuple[set[int], set[Definition], list[dict[str, Any]]]:
    guard_branches: set[int] = set()
    guard_definitions: set[Definition] = set()
    evidence: list[dict[str, Any]] = []
    trap_memo: dict[int, bool] = {}
    for branch_index, effect in enumerate(effects):
        if (
            branch_index not in reachable
            or effect.branch_kind != "conditional"
            or len(successors[branch_index]) != 2
        ):
            continue
        trap_successors = [
            successor
            for successor in successors[branch_index]
            if _trap_only_path(
                successor, effects, successors, set(), trap_memo
            )
        ]
        if len(trap_successors) != 1:
            continue
        trap_successor = trap_successors[0]
        normal_successor = next(
            successor
            for successor in successors[branch_index]
            if successor != trap_successor
        )
        control_sources: set[Definition] = set()
        valid_control = True
        for use in effect.uses:
            if use.role != "control":
                continue
            definitions = _use_sources(incoming, branch_index, use)
            if len(definitions) != 1:
                valid_control = False
                break
            control_sources.update(definitions)
        if not valid_control or not control_sources:
            continue

        compared_sources: set[Definition] = set()
        for definition in control_sources:
            if definition.instruction < 0:
                compared_sources.add(definition)
                continue
            producer = effects[definition.instruction]
            if definition.register != "nzcv":
                compared_sources.add(definition)
                continue
            compare_uses = [
                use for use in producer.uses if use.role == "compare"
            ]
            if not compare_uses:
                valid_control = False
                break
            for compare_use in compare_uses:
                sources = _use_sources(
                    incoming, definition.instruction, compare_use
                )
                if len(sources) != 1:
                    valid_control = False
                    break
                compared_sources.update(sources)
        if not valid_control or not compared_sources:
            continue

        guarded_access: tuple[int, str, str] | None = None
        for memory_index, memory_effect in enumerate(effects):
            if (
                memory_index not in reachable
                or not memory_effect.memory
                or normal_successor not in dominators[memory_index]
                or branch_index not in dominators[memory_index]
            ):
                continue
            for address_use in memory_effect.uses:
                if address_use.role != "address":
                    continue
                sources = _use_sources(
                    incoming, memory_index, address_use
                )
                if len(sources) == 1 and compared_sources & set(sources):
                    guarded_access = (
                        memory_index,
                        address_use.register,
                        next(iter(sources)).register,
                    )
                    break
            if guarded_access is not None:
                break
        if guarded_access is None:
            continue

        guard_branches.add(branch_index)
        guard_definitions.update(control_sources)
        evidence.append(
            {
                "branch_instruction_index": branch_index,
                "trap_successor_index": trap_successor,
                "normal_successor_index": normal_successor,
                "guarded_memory_instruction_index": guarded_access[0],
                "guarded_address_register": guarded_access[1],
                "compared_definition_register": guarded_access[2],
            }
        )
    return guard_branches, guard_definitions, evidence


def _build_consumers(
    effects: Sequence[Effect],
    incoming: Sequence[dict[str, frozenset[Definition]]],
    reachable: set[int],
) -> dict[Definition, list[Consumer]]:
    consumers: dict[Definition, list[Consumer]] = defaultdict(list)
    for index, effect in enumerate(effects):
        if index not in reachable:
            continue
        for use in effect.uses:
            definitions = _use_sources(incoming, index, use)
            ambiguous = len(definitions) != 1
            for definition in definitions:
                consumers[definition].append(
                    Consumer(index, use, ambiguous)
                )
    return consumers


def _trace_definition(
    definition: Definition,
    *,
    instructions: Sequence[Instruction],
    effects: Sequence[Effect],
    consumers: Mapping[Definition, Sequence[Consumer]],
    structural_guard_branches: set[int],
    structural_guard_definitions: set[Definition],
    stack: set[Definition],
    memo: dict[Definition, Trace],
) -> Trace:
    if definition in memo:
        cached = memo[definition]
        return Trace(
            roles=set(cached.roles),
            unknown_reasons=set(cached.unknown_reasons),
            evidence=list(cached.evidence),
        )
    if definition in stack:
        return Trace(unknown_reasons={"cyclic producer-consumer path"})
    uses = consumers.get(definition, ())
    if not uses:
        return Trace(unknown_reasons={"definition has no proven consumer"})
    next_stack = set(stack)
    next_stack.add(definition)
    result = Trace()
    for consumer in uses:
        instruction = instructions[consumer.instruction]
        if consumer.ambiguous:
            result.unknown_reasons.add(
                "consumer has multiple reaching definitions at a CFG join"
            )
            result.evidence.append(
                {
                    "offset": instruction.offset,
                    "instruction": instruction.text,
                    "role": "ambiguous_join",
                }
            )
            continue
        role = consumer.use.role
        if role == "address":
            result.roles.add("address")
        elif role == "escape":
            result.unknown_reasons.add(
                "value escapes through untyped memory or a signatureless return"
            )
        elif role == "control":
            if consumer.instruction in structural_guard_branches:
                result.roles.add("structural_guard")
            else:
                result.roles.add("control")
        elif role == "compare":
            if (
                Definition(consumer.instruction, "nzcv")
                in structural_guard_definitions
            ):
                result.roles.add("structural_guard")
            else:
                result.roles.add("control")
        elif role == "unknown":
            result.unknown_reasons.add(
                "value reaches an opaque instruction or call boundary"
            )
        elif role != "value":
            result.unknown_reasons.add(f"unrecognized consumer role {role}")

        result.evidence.append(
            {
                "offset": instruction.offset,
                "instruction": instruction.text,
                "role": role,
            }
        )
        propagation_targets = [
            target_register
            for target_register in consumer.use.propagate_to
            if target_register in effects[consumer.instruction].definitions
        ]
        if len(propagation_targets) != len(consumer.use.propagate_to):
            result.unknown_reasons.add(
                "producer-consumer propagation target is not defined"
            )
        active_targets = [
            target_register
            for target_register in propagation_targets
            if consumers.get(
                Definition(consumer.instruction, target_register)
            )
        ]
        if propagation_targets and not active_targets:
            result.unknown_reasons.add(
                "value transformer has no proven live output"
            )
        for target_register in active_targets:
            target = Definition(consumer.instruction, target_register)
            result.merge(
                _trace_definition(
                    target,
                    instructions=instructions,
                    effects=effects,
                    consumers=consumers,
                    structural_guard_branches=structural_guard_branches,
                    structural_guard_definitions=structural_guard_definitions,
                    stack=next_stack,
                    memo=memo,
                )
            )
    memo[definition] = Trace(
        roles=set(result.roles),
        unknown_reasons=set(result.unknown_reasons),
        evidence=list(result.evidence),
    )
    return result


def _category(trace: Trace) -> tuple[str, str]:
    if trace.unknown_reasons:
        return (
            "unknown",
            "; ".join(sorted(trace.unknown_reasons)),
        )
    semantic_groups: set[str] = set()
    if "address" in trace.roles:
        semantic_groups.add("address")
    if "structural_guard" in trace.roles:
        semantic_groups.add("structural_guard")
    if "data" in trace.roles:
        semantic_groups.add("algorithmic")
    if "control" in trace.roles:
        semantic_groups.add("control")
    if not semantic_groups:
        return "unknown", "no terminal semantic consumer was proven"
    if semantic_groups == {"control"}:
        return (
            "unknown",
            "control semantics are not proven bounds/check or algorithmic data",
        )
    if len(semantic_groups) > 1:
        return (
            "mixed",
            "producer has proven consumers in multiple semantic categories: "
            + ", ".join(sorted(semantic_groups)),
        )
    only = next(iter(semantic_groups))
    if only == "address":
        return (
            "address_generation",
            "all proven producer-consumer paths terminate in memory-address operands",
        )
    if only == "structural_guard":
        return (
            "structural_address_guard",
            "flags/control structurally guard a dominated eventual address "
            "and the other complete CFG path reaches an architectural trap; "
            "linear-memory limit or removable-check semantics are not proven",
        )
    return (
        "algorithmic_alu",
        "all terminal uses have explicit typed non-address data provenance",
    )


def analyze_instruction_stream(
    instructions: Sequence[Any],
    *,
    broad_classes: Sequence[str],
    samples_by_offset: Mapping[int, int],
    total_run_samples: int,
    address_base: int = 0,
    global_attributed_samples: int | None = None,
) -> dict[str, Any]:
    if total_run_samples <= 0:
        raise ProvenanceError("total_run_samples must be positive")
    if global_attributed_samples is None:
        global_attributed_samples = total_run_samples
    if not 0 <= global_attributed_samples <= total_run_samples:
        raise ProvenanceError(
            "global_attributed_samples must be within total_run_samples"
        )
    normalized = normalize_instructions(instructions, address_base=address_base)
    if len(broad_classes) != len(normalized):
        raise ProvenanceError(
            "broad instruction classes must align exactly with disassembly"
        )
    known_offsets = {instruction.offset for instruction in normalized}
    invalid_sample_offsets = []
    for offset, samples in samples_by_offset.items():
        try:
            count = int(samples)
        except (TypeError, ValueError) as exc:
            raise ProvenanceError(
                f"sample count at offset {offset!r} is not an integer"
            ) from exc
        if offset not in known_offsets or count < 0:
            invalid_sample_offsets.append(offset)
    invalid_sample_offsets.sort(key=str)
    if invalid_sample_offsets:
        raise ProvenanceError(
            "sample mappings must use exact nonnegative instruction offsets; "
            f"invalid offsets: {invalid_sample_offsets[:8]}"
        )
    effects = [instruction_effect(instruction) for instruction in normalized]
    successors, predecessors, external_targets = build_cfg(normalized, effects)
    reachable = _reachable(successors)
    incoming, _ = _reaching_definitions(effects, predecessors, reachable)
    dominators = _dominators(predecessors, reachable)
    structural_guard_branches, structural_guard_definitions, guard_evidence = (
        _structural_address_guard_branches(
            effects,
            successors,
            incoming,
            dominators,
            reachable,
        )
    )
    consumers = _build_consumers(effects, incoming, reachable)
    memo: dict[Definition, Trace] = {}
    decision_cache: dict[int, tuple[str, str, Trace]] = {}

    def classify_index(index: int) -> tuple[str, str, Trace]:
        if index in decision_cache:
            return decision_cache[index]
        instruction = normalized[index]
        effect = effects[index]
        if index not in reachable:
            category = "unknown"
            reason = "instruction is unreachable in the recovered CFG"
            trace = Trace(unknown_reasons={reason})
        elif not effect.recognized:
            category = "unknown"
            reason = "instruction semantics are not recognized conservatively"
            trace = Trace(unknown_reasons={reason})
        else:
            definitions = {
                register
                for register in (
                    effect.primary_definitions
                    | effect.secondary_definitions
                )
                if consumers.get(Definition(index, register))
            }
            if not definitions:
                definitions = set(effect.primary_definitions)
            combined = Trace()
            for register in definitions:
                combined.merge(
                    _trace_definition(
                        Definition(index, register),
                        instructions=normalized,
                        effects=effects,
                        consumers=consumers,
                        structural_guard_branches=structural_guard_branches,
                        structural_guard_definitions=structural_guard_definitions,
                        stack=set(),
                        memo=memo,
                    )
                )
            trace = combined
            category, reason = _category(trace)
        decision_cache[index] = (category, reason, trace)
        return decision_cache[index]

    def aggregate(indices: Sequence[int], *, share_name: str) -> dict[str, Any]:
        static_counts = Counter()
        sample_counts = Counter()
        reason_counts = Counter()
        sampled_instructions: list[dict[str, Any]] = []
        for index in indices:
            instruction = normalized[index]
            category, reason, trace = classify_index(index)
            static_counts[category] += 1
            samples = int(samples_by_offset.get(instruction.offset, 0))
            sample_counts[category] += samples
            reason_counts[(category, reason)] += 1
            if samples:
                sampled_instructions.append(
                    {
                        "offset": instruction.offset,
                        "address": instruction.address,
                        "instruction": instruction.text,
                        "samples": samples,
                        "percent_of_run": (
                            100.0 * samples / total_run_samples
                        ),
                        "category": category,
                        "reason": reason,
                        "path_evidence": trace.evidence[:12],
                    }
                )
        universe_samples = sum(sample_counts.values())
        proven_samples = sum(
            sample_counts[category] for category in PROVEN_CATEGORIES
        )
        sampled_instructions.sort(
            key=lambda item: (-item["samples"], item["offset"])
        )
        return {
            "samples": universe_samples,
            "partition_samples": sum(sample_counts.values()),
            "static_instructions": len(indices),
            "categories": {
                category: {
                    "samples": sample_counts[category],
                    "percent_of_run": (
                        100.0
                        * sample_counts[category]
                        / total_run_samples
                    ),
                    share_name: (
                        100.0
                        * sample_counts[category]
                        / universe_samples
                        if universe_samples
                        else 0.0
                    ),
                    "static_instructions": static_counts[category],
                }
                for category in CATEGORIES
            },
            "coverage": {
                "proven_samples": proven_samples,
                "proven_percent": (
                    100.0 * proven_samples / universe_samples
                    if universe_samples
                    else 0.0
                ),
                "unknown_samples": sample_counts["unknown"],
                "unknown_percent_of_run": (
                    100.0
                    * sample_counts["unknown"]
                    / total_run_samples
                ),
                "mixed_samples": sample_counts["mixed"],
                "mixed_percent_of_run": (
                    100.0 * sample_counts["mixed"] / total_run_samples
                ),
            },
            "sampled_instructions": sampled_instructions,
            "reason_counts": [
                {
                    "category": category,
                    "reason": reason,
                    "static_instructions": count,
                }
                for (category, reason), count in sorted(
                    reason_counts.items(),
                    key=lambda item: (item[0][0], item[0][1]),
                )
            ],
        }

    legacy_indices = [
        index
        for index, broad_class in enumerate(broad_classes)
        if broad_class == "alu"
    ]
    common_indices = [
        index
        for index, (broad_class, effect) in enumerate(
            zip(broad_classes, effects)
        )
        if broad_class == "alu" or effect.gating_candidate
    ]
    legacy = aggregate(
        legacy_indices, share_name="percent_of_broad_alu"
    )
    common = aggregate(
        common_indices, share_name="percent_of_common_universe"
    )
    if legacy["partition_samples"] != legacy["samples"]:
        raise ProvenanceError(
            "legacy ALU sample partitions do not reconcile"
        )
    if common["partition_samples"] != common["samples"]:
        raise ProvenanceError(
            "common gating-universe sample partitions do not reconcile"
        )
    mapped_instruction_samples = sum(
        int(samples) for samples in samples_by_offset.values()
    )
    excluded_instruction_samples = (
        mapped_instruction_samples - common["samples"]
    )
    if excluded_instruction_samples < 0:
        raise ProvenanceError(
            "common gating universe exceeds mapped instruction samples"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": ANALYSIS_KIND,
        "status": "measured",
        "broad_class": "all_alu",
        "total_run_samples": total_run_samples,
        "broad_alu_samples": legacy["samples"],
        "broad_alu_percent_of_run": (
            100.0 * legacy["samples"] / total_run_samples
        ),
        "partition_samples": legacy["partition_samples"],
        "instruction_count": len(normalized),
        "broad_alu_static_instructions": legacy["static_instructions"],
        "categories": legacy["categories"],
        "coverage": {
            **legacy["coverage"],
            "proven_percent_of_broad_alu": legacy["coverage"][
                "proven_percent"
            ],
        },
        "common_gating_universe": common,
        "instruction_sample_accounting": {
            "mapped_instruction_samples": mapped_instruction_samples,
            "common_candidate_samples": common["samples"],
            "conclusively_excluded_non_candidate_samples": (
                excluded_instruction_samples
            ),
        },
        "global_sample_mapping": {
            "attributed_samples": global_attributed_samples,
            "unattributed_samples": (
                total_run_samples - global_attributed_samples
            ),
            "total_samples": total_run_samples,
        },
        "cfg": {
            "reachable_instructions": len(reachable),
            "external_branch_targets": external_targets,
            "structural_address_guard_branches": len(
                structural_guard_branches
            ),
            "structural_address_guard_evidence": [
                {
                    **item,
                    "branch_offset": normalized[
                        item["branch_instruction_index"]
                    ].offset,
                    "branch_instruction": normalized[
                        item["branch_instruction_index"]
                    ].text,
                    "guarded_memory_offset": normalized[
                        item["guarded_memory_instruction_index"]
                    ].offset,
                    "guarded_memory_instruction": normalized[
                        item["guarded_memory_instruction_index"]
                    ].text,
                }
                for item in guard_evidence
            ],
        },
        "sampled_instructions": legacy["sampled_instructions"],
        "reason_counts": legacy["reason_counts"],
        "soundness": {
            "register_aliases": (
                "Wn and Xn share one architectural value; W writes replace "
                "the full tracked value because AArch64 zero-extends them."
            ),
            "joins_and_loops": (
                "A use with multiple reaching definitions, or a cyclic path "
                "without a proven terminal role, remains unknown."
            ),
            "calls": (
                "AAPCS64 possible x0-x7 arguments are opaque uses; x0-x18, "
                "x30, and NZCV are clobbered. No engine-specific register "
                "meaning is assumed."
            ),
            "structural_address_guard": (
                "A structural guard requires a complete CFG trap path plus a "
                "dominated eventual address sharing the compared definition. "
                "It does not prove a linear-memory limit, engine bounds check, "
                "or removable check."
            ),
            "mixed": (
                "Only independently proven semantic roles can produce mixed. "
                "Untyped stores, signatureless returns, and unproven control "
                "remain unknown escapes instead."
            ),
        },
    }


def unavailable_analysis(reason: str, *, total_run_samples: int | None = None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": ANALYSIS_KIND,
        "status": "unavailable",
        "reason": reason,
        "total_run_samples": total_run_samples,
    }


def compare_engine_analyses(
    wamr: Mapping[str, Any],
    wasmtime: Mapping[str, Any],
    *,
    threshold_pct: float = DEFAULT_GATE_THRESHOLD_PCT,
) -> dict[str, Any]:
    if threshold_pct < 0:
        raise ProvenanceError("gate threshold must be nonnegative")
    unavailable = [
        f"{name}: {analysis.get('reason', 'analysis unavailable')}"
        for name, analysis in (("wamr", wamr), ("wasmtime", wasmtime))
        if analysis.get("status") != "measured"
    ]
    if unavailable:
        return {
            "threshold_percentage_points": threshold_pct,
            "status": "blocked",
            "optimization_authorized": False,
            "reason": "; ".join(unavailable),
            "categories": {},
        }
    for name, analysis in (("wamr", wamr), ("wasmtime", wasmtime)):
        if analysis.get("partition_samples") != analysis.get("broad_alu_samples"):
            raise ProvenanceError(
                f"{name} narrow partitions do not reconcile to broad ALU"
            )
        common = analysis.get("common_gating_universe")
        if (
            not isinstance(common, Mapping)
            or common.get("partition_samples") != common.get("samples")
        ):
            raise ProvenanceError(
                f"{name} lacks a reconciled complete common gating universe"
            )
    wasmtime_common = wasmtime["common_gating_universe"]
    wasmtime_global_unattributed = wasmtime.get(
        "global_sample_mapping", {}
    ).get("unattributed_samples")
    if not isinstance(wasmtime_global_unattributed, int):
        raise ProvenanceError(
            "wasmtime global unattributed sample count is missing"
        )
    wasmtime_instruction_unresolved = wasmtime.get(
        "sample_mapping", {}
    ).get("unresolved_function_samples", 0)
    wasmtime_function_mapped = wasmtime.get(
        "sample_mapping", {}
    ).get("mapped_function_samples", 0)
    if (
        not isinstance(wasmtime_instruction_unresolved, int)
        or wasmtime_instruction_unresolved < 0
        or not isinstance(wasmtime_function_mapped, int)
        or wasmtime_function_mapped < 0
    ):
        raise ProvenanceError(
            "wasmtime function sample mapping is invalid"
        )
    if (
        wasmtime_function_mapped + wasmtime_instruction_unresolved
        > wasmtime["global_sample_mapping"]["attributed_samples"]
    ):
        raise ProvenanceError(
            "wasmtime function-mapped samples exceed globally attributed samples"
        )
    wasmtime_unknown_upper = (
        wasmtime_common["categories"]["unknown"]["percent_of_run"]
        + wasmtime_common["categories"]["mixed"]["percent_of_run"]
        + (
            100.0
            * (
                wasmtime_instruction_unresolved
                + wasmtime_global_unattributed
            )
            / wasmtime["total_run_samples"]
        )
    )
    categories = {}
    for category in GATE_CATEGORIES:
        wamr_category = wamr["common_gating_universe"]["categories"][
            category
        ]
        wasmtime_category = wasmtime_common["categories"][category]
        wamr_share = wamr_category["percent_of_run"]
        wasmtime_share = wasmtime_category["percent_of_run"]
        conservative_wasmtime_upper = wasmtime_share + wasmtime_unknown_upper
        conservative_headroom = wamr_share - conservative_wasmtime_upper
        categories[category] = {
            "wamr_samples": wamr_category["samples"],
            "wasmtime_samples": wasmtime_category["samples"],
            "wamr_percent_of_run": wamr_share,
            "wasmtime_percent_of_run": wasmtime_share,
            "observed_delta_percentage_points": wamr_share - wasmtime_share,
            "wasmtime_unknown_mixed_instruction_unresolved_or_global_unattributed_upper_percentage_points": (
                wasmtime_unknown_upper
            ),
            "wasmtime_instruction_unresolved_samples": (
                wasmtime_instruction_unresolved
            ),
            "wasmtime_global_unattributed_samples": (
                wasmtime_global_unattributed
            ),
            "conservative_headroom_percentage_points": conservative_headroom,
            "clears_threshold": conservative_headroom >= threshold_pct,
        }
    cleared = [
        category
        for category, values in categories.items()
        if values["clears_threshold"]
    ]
    return {
        "threshold_percentage_points": threshold_pct,
        "status": "cleared" if cleared else "not-cleared",
        "optimization_authorized": bool(cleared),
        "cleared_categories": cleared,
        "reason": (
            "at least one shared proven category clears the conservative "
            "run-wide threshold"
            if cleared
            else "no shared proven category clears the conservative run-wide "
            "threshold after treating Wasmtime common-universe unknown/mixed, "
            "instruction-unresolved, and globally unattributed samples as a "
            "possible upper bound"
        ),
        "categories": categories,
    }
