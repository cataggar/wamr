---
name: aot-diff-debug
description: Debug AOT codegen mismatches by comparing wamr-AOT vs the wasmtime reference oracle on real wasm binaries. Use when AOT produces wrong results, CRC errors, garbled output, or any computation mismatch versus wasmtime.
---
# AOT Differential Debugging

Systematically find AOT codegen bugs by comparing wasmtime (the
reference oracle) and wamr's AOT execution of the same wasm binary.
This skill encodes the lessons from debugging the CoreMark CRC bug
(memory_fill r11 clobber, issue #109) and the `String.prototype.slice`
StarlingMonkey regression (#754).

## When to Use

- AOT produces wrong numeric results, garbled text, or CRC errors
- A wasm program works under wasmtime but fails under wamr's AOT
- You suspect a codegen bug but don't know which instruction is wrong

## Core Principle

**Use wasmtime as an oracle. Diff real outputs first, hypothesize second.**

Do NOT start by:
- Reading codegen source code for suspicious patterns
- Building synthetic differential tests for hypothesized instructions
- Auditing register allocation or spill/reload logic

DO start by:
- Running the same binary through both runtimes and comparing output

## Workflow

### Phase 1: Reproduce and Diff (do this FIRST, every time)

Use the built-in `wamrc verify` subcommand (#757). It compiles the
wasm to AOT, runs it under both `wamr` and `wasmtime`, and prints
the first-divergence offset with hex+ASCII context on each side:

```sh
./zig-out/bin/wamrc verify program.wasm
```

Exit codes:
- `0` — outputs match
- `1` — divergence found (the printed report names the offset)
- `2` — setup error (e.g. wasmtime missing)

For a guest-arg-sensitive case:

```sh
./zig-out/bin/wamrc verify program.wasm -- arg1 arg2
```

To also mount host directories or pass env vars (translated to the
right flag spelling for each runtime):

```sh
./zig-out/bin/wamrc verify program.wasm --map-dir /host/data::/work --env LOG=trace
```

Run `wamrc verify help` for the full flag list (timeouts, JSON
output for scripting, `--keep-cwasm` for post-mortem, etc.).

**Characterize the divergence** from the report:
- Is it a wrong number? (codegen arithmetic bug)
- Is it garbled characters? (memory corruption or wrong pointer)
- Is it extra/missing output? (control flow bug)
- Is it a specific character that's always wrong? (lookup table or
  constant bug)

#### Fallback when wasmtime isn't available

If wasmtime isn't on the box, run the wamr arm directly and diff
against a previously-known-good capture from another runtime
(wasmtime on another machine, or a known good wamr build for a
regression bisect):

```sh
./zig-out/bin/wamrc run program.wasm > wamr-out.txt 2>&1
diff wamr-out.txt golden-out.txt
```

Install wasmtime instead — `wamrc verify` is the right tool here —
this fallback is for environments where that isn't possible.

### Phase 2: Build a Minimal Reproducer

Once you know WHAT is wrong from Phase 1, build the smallest C
program that reproduces it.

**Key technique**: compile C to wasm32-wasi, run `wamrc verify`
again on the minimal program:

```sh
zig build-exe test.c -target wasm32-wasi -lc -O ReleaseFast --name test
./zig-out/bin/wamrc verify test.wasm
```

**Minimization strategy** (in order):
1. Remove unrelated functionality from the C source
2. Replace complex formatting with `putchar()` to isolate whether values or formatting is wrong
3. Add both `%d` (decimal) and `%x` (hex) output to distinguish value errors from formatting errors
4. Test edge cases: value 0, negative values, max values, boundary values

**Example**: if hex output is garbled but decimal output is correct, the BUG is in how the hex formatter works (memory_fill, lookup table, etc.), not in the value computation.

### Phase 3: Identify the Guilty Instruction Handler

Once you have a minimal reproducer (ideally <10 lines of C):

1. **Enable IR dump** in `src/compiler/codegen/x86_64/compile.zig`:
   ```zig
   const dump_all: bool = true;  // line ~1300
   ```
   Rebuild `wamrc` and compile the test wasm. The IR dump shows every function's instructions and register allocations.

2. **Search the IR dump** for the operation that produces wrong results:
   - `memory_fill` — REP STOSB fill operations
   - `memory_copy` — REP MOVSB copy operations
   - `div_u` / `rem_u` — division and remainder
   - `call_indirect` — indirect function calls
   - `br_table` — switch/jump tables
   - `select` — conditional select

3. **Check the instruction handler** in `compile.zig` for scratch register clobbers:
   - Does it stash a value in `r10` or `r11` before calling a helper?
   - Does the helper (`emitMemBoundsCheck`, `emitMemBoundsCheckDynamic`, `emitOobCmpAndTrap`, `emitCallIndirectSigCheck`) clobber that register?
   - **Known clobber sets**:
     - `emitMemBoundsCheck` / `emitMemBoundsCheckDynamic`: clobbers **r11**, r10, rax, param_regs[0]
     - `emitOobCmpAndTrap`: clobbers r10→param_regs[0], rax
     - `emitCallIndirectSigCheck`: clobbers **rcx**, **r11**

### Phase 4: Fix and Verify

1. **Fix pattern**: prefer "load late" over "stash and restore":
   ```zig
   // BAD: stash in r11, call helper that clobbers r11, restore from r11
   try code.movRegReg(.r11, .rax);   // save
   try emitMemBoundsCheckDynamic();   // CLOBBERS r11!
   try code.movRegReg(.rax, .r11);   // "restore" — WRONG

   // GOOD: load the value AFTER the helper call
   try emitMemBoundsCheckDynamic();   // do bounds check first
   const val_reg = try useVReg(...);  // load value after (survives check)
   ```

2. **Run unit tests**: `zig build test`
3. **Re-run the diff**: `./zig-out/bin/wamrc verify test.wasm` — must
   exit 0 (outputs match)
4. **Re-run the original failing program** (e.g., CoreMark) through
   `wamrc verify` — outputs must match
5. **Run spec tests**: `zig build spec-tests-aot` (check for regressions)

## Anti-Patterns (things that waste time)

| Anti-Pattern | Why It Wastes Time | Do Instead |
|---|---|---|
| Reading codegen handlers looking for bugs | Too many handlers, bug could be anywhere | Diff real output with `wamrc verify` first |
| Building synthetic wasm tests for hypothesized opcodes | Tests pass because they don't exercise the real pattern | Build minimal C reproducer of the actual failure |
| Testing with different register counts | Only eliminates allocatable-reg bugs, not scratch-reg bugs | Check scratch register clobbers in handlers |
| Assuming "spec tests pass ∴ all opcodes correct" | Spec tests don't test libc internals or complex interactions | Test with real programs |
| Hypothesizing before observing | Leads to confirmation bias | Always observe the divergence first |
| Hand-building probe scripts to find the divergence | Hours of wiring vs one command | Use `wamrc verify` — that's literally what it's for (#754 spent 6 hours pre-verify, the actual fix was 13 lines) |

## Historical Bugs Found with This Approach

### CoreMark CRC bug (issue #109)
- **Symptom**: list CRC=0x5531 (expected 0xe714), state CRC=0x344b (expected 0x8e3a)
- **Phase 1 finding**: hex digit '0' replaced with garbage characters (garbled printf output)
- **Phase 2 reproducer**: `printf("%d\n", 0)` — 1 line of C
- **Root cause**: `memory_fill` handler stashed fill value in r11, then called `emitMemBoundsCheckDynamic` which clobbers r11 via `lea r11, [rax+rcx]`
- **Fix**: load fill value AFTER bounds check (commit cfa702b5)
- **Time wasted before this approach**: ~8 hours of hypothesis-driven debugging
- **Time with this approach**: <1 hour

### `String.prototype.slice` regression (issue #754)
- **Symptom**: jco-componentize-built JS guests returned `""` from
  `"hello".slice(0, 3)` (expected `"hel"`) under wamr-AOT; wasmtime
  was correct.
- **Phase 1 (if `wamrc verify` had existed)**: one command — would
  have shown the empty-string divergence at the first `slice`-shaped
  line of `Math.min(3, 11) =` output. Bisect to the
  `frontend.zig:lowerFunction` select-type fix-up's fixpoint loop
  follows in ~30 min.
- **Time wasted before this approach**: ~6 hours of probe-building.
- **Time with `wamrc verify`**: estimated <30 min. The fixture is
  preserved at `tests/regressions/754-slice/`.

## References

- `src/compiler/verify.zig` — `wamrc verify` orchestrator (#757)
- `src/compiler/verify_args.zig` — flag parser + per-runtime translation
- `src/compiler/codegen/x86_64/compile.zig` — x86-64 instruction handlers
- `src/compiler/codegen/x86_64/compile.zig:1298` — `dump_func_idx` / `dump_all` debug knobs
- `src/compiler/frontend.zig` — wasm-to-IR translation
- `src/compiler/codegen/x86_64/compile.zig:277` — `emitMemBoundsCheck` (clobbers r11)
- `src/compiler/codegen/x86_64/compile.zig:313` — `emitMemBoundsCheckDynamic` (clobbers r11)

## Companion skill

If after Phase 1 you've localized the bug to a specific function
(`local_func[N]` in a trap, or via diff narrowing) but it goes away
with `-O0` — i.e. the bug is somewhere in the IR optimization
pipeline rather than in codegen itself — hand off to the
**`aot-pass-bisect`** skill. That skill applies
`WAMR_AOT_SKIP_PASS=...:fn=<N>` + `--cache-dir` to localize the
responsible pass in ~5 cycles of seconds-each-when-cached
recompiles, instead of the 8-40 min per cycle a full recompile
takes on large components.
