# Mutation testing

`scripts/mutation_test.py` is a small one-shot mutation-testing helper for Zig
source files. It temporarily applies one mutant, runs a test command, records
whether the tests failed, restores the original file, and continues with the
next mutant.

The script has no runtime dependencies beyond Python 3 and the existing Zig
toolchain.

## Quick start

Run from the repository root:

```sh
unset ZIG_LOCAL_CACHE_DIR
export ZIG_GLOBAL_CACHE_DIR="$PWD/.zig-global-cache"
python3 scripts/mutation_test.py src/compiler/ir/verifier.zig \
  --seed 737 \
  --limit 100 \
  --timeout 20 \
  --test-cmd 'unset ZIG_LOCAL_CACHE_DIR; export ZIG_GLOBAL_CACHE_DIR="$PWD/.zig-global-cache"; zig build test --summary all 2>&1 | tail -5' \
  --out docs/mutation-testing/reports/verifier.md
```

Useful options:

- `--seed N`: shuffle mutants deterministically before applying `--limit`.
- `--limit N`: execute at most `N` mutants.
- `--line-range START:END`: restrict mutations to inclusive source ranges;
  repeat the option to cover multiple functions.
- `--timeout SECONDS`: kill a mutant test command that appears to hang.
- `--test-cmd CMD`: shell command used to classify each mutant.
- `--out PATH`: markdown report path.
- `--list`: show selected mutants without running tests.

The default test command is the full `zig build test --summary all` suite. For
local investigation, it is often faster to use `zig test <target.zig>` first and
rerun interesting survivors with the full suite before adding regression tests.

## Mutators

Each mutant applies exactly one source edit:

1. Comment out single-line `clearAll(...)` calls whose line ends in `;`.
2. Swap relational operators: `<` with `>`, and `<=` with `>=`.
3. Delete standalone `continue;`, `break;`, and `return;` statements.
4. Flip integer literals `0` and `1` in expression-like contexts.
5. Flip off-by-one operations: `+ 1` with `- 1`.

The mutators are intentionally conservative and line-oriented. They skip
multi-line `clearAll` calls and avoid introducing extra dependencies or a Zig
parser.

## Reading reports

Reports are markdown tables sorted with `SURVIVED` mutants first:

- `KILLED`: the test command returned non-zero, so existing tests noticed the
  behavior change or the mutant failed to compile.
- `SURVIVED`: the test command returned zero, so the selected tests did not
  detect the behavior change.
- Timed-out mutants are treated as killed because the command is non-zero.

Survivors are not automatically bugs. They are prompts for focused test review:
check whether the mutated line changes a meaningful behavior, then add or extend
a test that would fail for that mutant.

## Initial reports

Initial compiler/IR reports live in `docs/mutation-testing/reports/`:

- `passes.md`: sampled load-forwarding and nearby pass code ranges.
- `forward_redundant_loads_dominator.md`: exhaustive run for the file.
- `verifier.md`: seeded sample capped with `--limit 100`.

The reports record the exact command, seed, limit, and line ranges used.

## Adding mutators

Add a new case in `line_mutants()` in `scripts/mutation_test.py` and return a
`Mutant` with the original line, mutated line, and a concise mutator name.
Prefer edits that are:

- single-line and easy to restore;
- deterministic;
- syntactically plausible for Zig;
- specific enough to produce actionable survivors.

Validate new mutators with `--list`, then run a tiny `--limit` sample before a
larger report. Always verify `git diff` after interruption to confirm the target
file was restored.
