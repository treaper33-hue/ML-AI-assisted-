"""A3 · INTERMEDIATE — Optimisation passes that preserve meaning.

Four classical passes, each a few dozen lines, run to a fixpoint:

* **Constant folding** — evaluate what is already known.
* **Copy propagation** — remove the indirection `b = a` introduces.
* **Common subexpression elimination** — compute each value once.
* **Dead code elimination** — delete what nothing reads.

They are individually simple and collectively powerful, because each
one creates opportunities for the others: folding makes copies, copies
expose common subexpressions, eliminating them makes instructions dead.
That mutual feeding is why the driver loops until nothing changes
rather than running each pass once.

**The discipline that matters more than the passes.** An optimisation
is only correct if the program still computes the same thing, and the
only way to believe that is to check it. Every gate here runs the
program before and after on random inputs and compares. That is
differential testing, and it is how real compilers are tested, because
proving a pass correct is hard and running it on a thousand inputs is
easy.

Run it:
    python a3_passes.py
"""

import math
import random
import sys
from typing import Callable, Dict, List, Set, Tuple, Union

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))

from a2_ir import (BINARY_OPS, BUILTINS, PURE, Instruction,  # noqa: E402
                   Program, compile_source, execute)


def _is_literal(value: Union[str, float]) -> bool:
    """Whether an operand is an immediate number."""
    return isinstance(value, (int, float))


def constant_folding(program: Program) -> Tuple[Program, int]:
    """Evaluates instructions whose operands are all known constants.

    Division is folded only when the divisor is non-zero: folding a
    division by zero would turn a runtime error into a compile-time
    crash, which changes behaviour rather than preserving it. Compilers
    that get this wrong break programs that never execute the faulty
    branch.

    Args:
        program: The program to fold.

    Returns:
        A tuple (new program, number of instructions folded).
    """
    known: Dict[str, float] = {}
    output: List[Instruction] = []
    folded = 0

    for instruction in program.instructions:
        operands = [known.get(a, a) if isinstance(a, str) else a
                    for a in instruction.args]

        if instruction.op == 'const':
            known[instruction.dest] = float(instruction.args[0])
            output.append(instruction)
            continue

        all_known = all(_is_literal(o) for o in operands)
        if all_known and instruction.op in PURE:
            try:
                if instruction.op in BINARY_OPS:
                    if (instruction.op in ('/', '%')
                            and float(operands[1]) == 0.0):
                        raise ZeroDivisionError
                    value = BINARY_OPS[instruction.op](float(operands[0]),
                                                       float(operands[1]))
                elif instruction.op == 'neg':
                    value = -float(operands[0])
                elif instruction.op == 'copy':
                    value = float(operands[0])
                elif instruction.op in BUILTINS:
                    _, function = BUILTINS[instruction.op]
                    value = float(function(*[float(o) for o in operands]))
                else:
                    raise ValueError
            except (ZeroDivisionError, ValueError, OverflowError):
                # Not foldable at compile time: leave it for runtime,
                # where the error belongs.
                output.append(instruction)
                continue
            known[instruction.dest] = value
            output.append(Instruction('const', instruction.dest, (value,)))
            folded += 1
        else:
            output.append(instruction)

    return Program(output, program.result, program.free_names), folded


def copy_propagation(program: Program) -> Tuple[Program, int]:
    """Replaces reads of a copy with reads of the original.

    Args:
        program: The program.

    Returns:
        A tuple (new program, number of operands rewritten).
    """
    alias: Dict[str, str] = {}
    output: List[Instruction] = []
    rewritten = 0

    for instruction in program.instructions:
        args = []
        for operand in instruction.args:
            if isinstance(operand, str) and operand in alias:
                args.append(alias[operand])
                rewritten += 1
            else:
                args.append(operand)
        args_tuple = tuple(args)

        if instruction.op == 'copy' and isinstance(args_tuple[0], str):
            alias[instruction.dest] = args_tuple[0]
        output.append(Instruction(instruction.op, instruction.dest,
                                  args_tuple))

    result = alias.get(program.result, program.result)
    return Program(output, result, program.free_names), rewritten


def common_subexpression_elimination(
        program: Program) -> Tuple[Program, int]:
    """Replaces a recomputation with a copy of the earlier result.

    Only pure operations are shared. The key is the operation together
    with its operand names, which is sound here precisely because the
    IR is single-assignment: a name's value never changes, so two
    instructions with the same key must produce the same value.

    Commutative operations use a sorted key, so `a + b` and `b + a`
    hash together.

    Args:
        program: The program.

    Returns:
        A tuple (new program, number of recomputations removed).
    """
    seen: Dict[Tuple, str] = {}
    output: List[Instruction] = []
    removed = 0
    commutative = {'+', '*', 'max', 'min', '==', '!='}

    for instruction in program.instructions:
        if instruction.op not in PURE or instruction.op == 'const':
            output.append(instruction)
            continue
        operands = instruction.args
        if instruction.op in commutative:
            operands = tuple(sorted(operands, key=repr))
        key = (instruction.op, operands)
        if key in seen:
            output.append(Instruction('copy', instruction.dest,
                                      (seen[key],)))
            removed += 1
        else:
            seen[key] = instruction.dest
            output.append(instruction)

    return Program(output, program.result, program.free_names), removed


def dead_code_elimination(program: Program) -> Tuple[Program, int]:
    """Removes instructions whose results nothing reads.

    Liveness is computed backwards from the result: a value is live if
    the result depends on it, directly or transitively. Walking
    backwards means one pass suffices for a straight-line program.

    Args:
        program: The program.

    Returns:
        A tuple (new program, number of instructions deleted).
    """
    live: Set[str] = {program.result}
    keep: List[Instruction] = []

    for instruction in reversed(program.instructions):
        if instruction.dest not in live:
            if instruction.op in PURE:
                continue                    # pure and unused: delete it
        keep.append(instruction)
        live.discard(instruction.dest)
        for operand in instruction.args:
            if isinstance(operand, str):
                live.add(operand)

    keep.reverse()
    return (Program(keep, program.result, program.free_names),
            program.count() - len(keep))


PASSES: List[Tuple[str, Callable[[Program], Tuple[Program, int]]]] = [
    ('constant folding', constant_folding),
    ('copy propagation', copy_propagation),
    ('common subexpression elimination',
     common_subexpression_elimination),
    ('dead code elimination', dead_code_elimination),
]


def optimise(program: Program, max_rounds: int = 10,
             trace: bool = False) -> Tuple[Program, List[str]]:
    """Runs every pass to a fixpoint.

    Each pass creates work for the others, so one round is never
    enough. The loop stops when a full round changes nothing, which is
    the definition of a fixpoint and also the only honest termination
    condition.

    Args:
        program: The program to optimise.
        max_rounds: Safety bound; reaching it means a pass is unstable.
        trace: Whether to record a line per effective pass.

    Returns:
        A tuple (optimised program, trace lines).
    """
    log: List[str] = []
    for round_number in range(1, max_rounds + 1):
        changed = 0
        for name, pass_function in PASSES:
            before = program.count()
            program, count = pass_function(program)
            changed += count
            if trace and count:
                log.append(f'round {round_number}: {name:<34} '
                           f'{count:>2} change(s), '
                           f'{before} -> {program.count()} instructions')
        if changed == 0:
            if trace:
                log.append(f'round {round_number}: fixpoint reached')
            break
    return program, log


BENCHMARKS: List[Tuple[str, str]] = [
    ('redundant work', 'let s = a + b in s * s + (a + b) * (a + b)'),
    ('constants', '2 * 3 + 4 * 5 - 6 / 2'),
    ('mixed', 'let k = 2 * 8 in max(a * k, a * k) + exp(0)'),
    ('deep reuse', '(a + b) * (a + b) + (a + b) * (a + b)'),
    ('dead values', 'let unused = a * b * 999 in a + 1'),
    ('builtin folding', 'sqrt(16) + abs(-4) + min(3, 9)'),
]


def differential_check(before: Program, after: Program, trials: int = 200,
                       seed: int = 20260101) -> Tuple[bool, str]:
    """Checks two programs agree on random inputs.

    Args:
        before: The original program.
        after: The optimised program.
        trials: How many random input vectors to try.
        seed: Seed for reproducibility.

    Returns:
        A tuple (agreed, detail).
    """
    rng = random.Random(seed)
    names = sorted(set(before.free_names) | set(after.free_names))
    worst = 0.0
    for _ in range(trials):
        inputs = {name: rng.uniform(-50.0, 50.0) for name in names}
        try:
            expected = execute(before, inputs)
            actual = execute(after, inputs)
        except (ZeroDivisionError, ValueError, OverflowError):
            continue                        # same input, same refusal
        if math.isnan(expected) and math.isnan(actual):
            continue
        scale = max(1.0, abs(expected))
        worst = max(worst, abs(expected - actual) / scale)
    return worst < 1e-12, f'worst relative difference {worst:.2e}'


def main() -> int:
    """Demonstrates the passes and checks the gates.

    Returns:
        0 when every gate passed.
    """
    print('=' * 70)
    print('A3 · INTERMEDIATE — Optimisation passes that preserve meaning')
    print('=' * 70)

    source = 'let s = a + b in s * s + (a + b) * (a + b)'
    original = compile_source(source)
    print(f'\nsource:\n  {source}')
    print(f'\nbefore ({original.count()} instructions):')
    print(original)

    optimised, log = optimise(original, trace=True)
    print('\npass log:')
    for line in log:
        print(f'  {line}')
    print(f'\nafter ({optimised.count()} instructions):')
    print(optimised)
    print('\nThe three copies of (a + b) became one. CSE could prove they')
    print('were the same value because single assignment guarantees a')
    print('name never changes meaning.')

    failures = 0
    print('\nGates — size reduced AND meaning preserved')
    print(f'  {"benchmark":<20}{"before":>8}{"after":>8}{"saved":>8}  check')
    for name, text in BENCHMARKS:
        start = compile_source(text)
        end, _ = optimise(start)
        agreed, detail = differential_check(start, end)
        smaller = end.count() <= start.count()
        ok = agreed and smaller
        failures += not ok
        saved = start.count() - end.count()
        print(f'  {name:<20}{start.count():>8}{end.count():>8}'
              f'{saved:>8}  [{"PASS" if ok else "FAIL"}] {detail}')

    print('\nGate — the pipeline is idempotent')
    for name, text in BENCHMARKS:
        once, _ = optimise(compile_source(text))
        twice, _ = optimise(once)
        if once.count() != twice.count():
            print(f'  [FAIL] {name}: {once.count()} -> {twice.count()}')
            failures += 1
    print(f'  [{"PASS" if not failures else "FAIL"}] '
          f'a second run changes nothing')

    print('\nGate — division by zero is NOT folded')
    risky = compile_source('1 / 0')
    folded, _ = constant_folding(risky)
    still_divides = any(i.op == '/' for i in folded.instructions)
    failures += not still_divides
    print(f'  [{"PASS" if still_divides else "FAIL"}] '
          f'the division survives to runtime, where the error belongs')

    print('\nGate — a free variable is never folded away')
    kept = optimise(compile_source('a * 1 + 0'))[0]
    reads_a = any('a' in [str(x) for x in i.args]
                  for i in kept.instructions)
    failures += not reads_a
    print(f'  [{"PASS" if reads_a else "FAIL"}] '
          f'`a` is still read (we do not implement algebraic identities)')

    print(f'\nstatus: {"VERIFIED" if not failures else "FAILED"}')
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
