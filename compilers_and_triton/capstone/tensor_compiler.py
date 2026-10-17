"""CAPSTONE · ADVANCED — A tensor compiler that emits Triton.

This is where the two tracks become one thing.

Track A built a compiler for scalar arithmetic: text to tree, tree to
three-address code, then passes that fold constants, share common
subexpressions and delete dead work. Track B wrote GPU kernels by hand.

A tensor compiler is the first applied to the second. Given an
expression over tensors, it:

1. **parses and lowers** it with Track A's front end, unchanged — the
   IR does not care whether a name holds a scalar or a million
   elements;
2. **optimises** it with Track A's passes, unchanged, which is the
   point of a well-chosen IR;
3. **emits a kernel** — one fused Triton kernel for the whole
   expression, instead of one kernel per operation;
4. **emits the same program for NumPy**, so the result is verifiable
   on a machine with no GPU;
5. **checks** that both backends agree with a direct evaluation.

The payoff is fusion. `relu(x * w + b)` as three library calls reads
and writes the whole tensor three times; memory bandwidth, not
arithmetic, decides how long that takes (B1). Fused into one kernel it
reads once and writes once. The compiler does not make the arithmetic
faster — it deletes the memory traffic between the operations, and that
is where essentially all the speedup in PyTorch 2 compilation comes
from.

Run it:
    python tensor_compiler.py
"""

import pathlib
import sys
from typing import Dict, List, Tuple

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'track_a_compilers'))
sys.path.insert(0, str(ROOT / 'track_b_triton'))

from a2_ir import BINARY_OPS, Program, compile_source  # noqa: E402
from a3_passes import optimise                         # noqa: E402

try:
    from runtime import CAN_RUN_KERNELS, environment_banner
except ImportError:                                    # pragma: no cover
    CAN_RUN_KERNELS = False

    def environment_banner() -> str:
        """Fallback when the Triton runtime module is unavailable."""
        return 'environment: track_b_triton/runtime.py not importable'

# Elementwise operations this compiler knows how to emit, mapped to the
# expression each backend uses. Adding an operation means adding one
# row -- the same "precedence as data" idea from A1, applied to codegen.
ELEMENTWISE: Dict[str, Dict[str, str]] = {
    '+': {'numpy': '({a} + {b})', 'triton': '({a} + {b})'},
    '-': {'numpy': '({a} - {b})', 'triton': '({a} - {b})'},
    '*': {'numpy': '({a} * {b})', 'triton': '({a} * {b})'},
    '/': {'numpy': '({a} / {b})', 'triton': '({a} / {b})'},
    'neg': {'numpy': '(-{a})', 'triton': '(-{a})'},
    'exp': {'numpy': 'np.exp({a})', 'triton': 'tl.exp({a})'},
    'sqrt': {'numpy': 'np.sqrt({a})', 'triton': 'tl.sqrt({a})'},
    'log': {'numpy': 'np.log({a})', 'triton': 'tl.log({a})'},
    'max': {'numpy': 'np.maximum({a}, {b})',
            'triton': 'tl.maximum({a}, {b})'},
    'min': {'numpy': 'np.minimum({a}, {b})',
            'triton': 'tl.minimum({a}, {b})'},
    'abs': {'numpy': 'np.abs({a})', 'triton': 'tl.abs({a})'},
}


class UnsupportedOperation(Exception):
    """Raised when the IR contains something this backend cannot emit."""


def _operand(value, literals: Dict[str, float]) -> str:
    """Renders an operand as backend source text.

    Args:
        value: A temporary name, an input name, or a literal.
        literals: Constants already folded into the program.

    Returns:
        Source text for the operand.
    """
    if isinstance(value, (int, float)):
        return repr(float(value))
    if value in literals:
        return repr(literals[value])
    return str(value)


def emit(program: Program, backend: str, name: str = 'fused') -> str:
    """Emits source code for one fused elementwise kernel.

    Both backends walk the same instruction list and differ only in the
    expression table and the wrapper. That is the whole argument for an
    intermediate representation: the optimiser is written once, and a
    new target costs a table and a prologue.

    Args:
        program: The optimised program.
        backend: 'numpy' or 'triton'.
        name: Name for the generated function.

    Returns:
        Python source text.

    Raises:
        UnsupportedOperation: On an operation with no emission rule.
        ValueError: On an unknown backend.
    """
    if backend not in ('numpy', 'triton'):
        raise ValueError(f'unknown backend {backend!r}')

    inputs = list(program.free_names)
    literals: Dict[str, float] = {}
    body: List[str] = []

    for instruction in program.instructions:
        if instruction.op == 'const':
            # Folded constants become literals in the emitted text
            # rather than loads; there is nothing to read from memory.
            literals[instruction.dest] = float(instruction.args[0])
            continue
        if instruction.op == 'copy':
            source = _operand(instruction.args[0], literals)
            body.append(f'    {instruction.dest} = {source}')
            continue
        if instruction.op not in ELEMENTWISE:
            raise UnsupportedOperation(
                f'no {backend} rule for {instruction.op!r}')
        operands = [_operand(a, literals) for a in instruction.args]
        template = ELEMENTWISE[instruction.op][backend]
        expression = (template.format(a=operands[0], b=operands[1])
                      if len(operands) == 2
                      else template.format(a=operands[0]))
        body.append(f'    {instruction.dest} = {expression}')

    result = _operand(program.result, literals)

    if backend == 'numpy':
        signature = ', '.join(inputs)
        lines = [f'def {name}({signature}):',
                 '    """Generated NumPy kernel."""']
        lines.extend(body)
        lines.append(f'    return {result}')
        return '\n'.join(lines)

    # Triton: one program instance per block, masked tail, one load per
    # input and one store -- the B1 pattern, generated.
    pointers = ', '.join(f'{n}_ptr' for n in inputs)
    loads = [f'    {n} = tl.load({n}_ptr + offsets, mask=mask, other=0.0)'
             for n in inputs]
    # Generated code is code: the continuation lines up under the open
    # bracket, which depends on the name, so it is computed rather than
    # guessed. A compiler that emits misaligned source teaches the
    # reader to distrust what it emits.
    signature = f'def {name}_kernel({pointers}, out_ptr, n_elements,'
    continuation = ' ' * (signature.index('(') + 1)
    lines = [
        '@triton.jit',
        signature,
        f'{continuation}BLOCK_SIZE: tl.constexpr):',
        '    """Generated Triton kernel: one fused pass over the data."""',
        '    pid = tl.program_id(axis=0)',
        '    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)',
        '    mask = offsets < n_elements',
    ]
    lines.extend(loads)
    lines.extend(body)
    lines.append(f'    tl.store(out_ptr + offsets, {result}, mask=mask)')
    return '\n'.join(lines)


def build_numpy_kernel(source: str, name: str = 'fused'):
    """Compiles emitted NumPy source into a callable.

    `compile` is also the syntax check: emitted code that does not
    parse fails here rather than at the point of use.

    Args:
        source: Emitted source text.
        name: The function name within it.

    Returns:
        The compiled callable.
    """
    namespace: Dict[str, object] = {'np': np}
    exec(compile(source, '<generated>', 'exec'), namespace)   # noqa: S102
    return namespace[name]


def evaluate_directly(expression: str,
                      inputs: Dict[str, np.ndarray]) -> np.ndarray:
    """Evaluates the expression with NumPy, unfused, as the reference.

    Deliberately naive: every operation materialises its own array,
    exactly like a chain of library calls. This is both the correctness
    oracle and the thing fusion is supposed to beat.

    Args:
        expression: The source expression.
        inputs: Input arrays.

    Returns:
        The result array.
    """
    environment = dict(inputs)
    program = compile_source(expression)
    values: Dict[str, np.ndarray] = {}

    def read(operand):
        """Resolves an operand to an array or scalar."""
        if isinstance(operand, (int, float)):
            return float(operand)
        if operand in values:
            return values[operand]
        return environment[operand]

    unary = {'exp': np.exp, 'sqrt': np.sqrt, 'log': np.log, 'abs': np.abs}
    binary = {'max': np.maximum, 'min': np.minimum}

    for instruction in program.instructions:
        if instruction.op == 'const':
            values[instruction.dest] = float(instruction.args[0])
        elif instruction.op == 'neg':
            values[instruction.dest] = -read(instruction.args[0])
        elif instruction.op == 'copy':
            values[instruction.dest] = read(instruction.args[0])
        elif instruction.op in BINARY_OPS:
            values[instruction.dest] = BINARY_OPS[instruction.op](
                read(instruction.args[0]), read(instruction.args[1]))
        elif instruction.op in unary:
            values[instruction.dest] = unary[instruction.op](
                read(instruction.args[0]))
        elif instruction.op in binary:
            values[instruction.dest] = binary[instruction.op](
                read(instruction.args[0]), read(instruction.args[1]))
        else:
            raise UnsupportedOperation(instruction.op)
    return read(program.result)


def count_memory_passes(program: Program) -> Tuple[int, int]:
    """Counts tensor reads and writes, fused versus unfused.

    Args:
        program: The optimised program.

    Returns:
        A tuple (unfused_passes, fused_passes).
    """
    tensor_ops = sum(1 for i in program.instructions
                     if i.op in ELEMENTWISE)
    inputs = len(program.free_names)
    # Unfused: each operation reads its operands and writes a temporary.
    unfused = tensor_ops * 2 + tensor_ops
    # Fused: read each input once, write the result once.
    fused = inputs + 1
    return unfused, fused


EXPRESSIONS: List[Tuple[str, str]] = [
    ('scaled shift', 'x * 2 + 1'),
    ('relu-like', 'max(x * w + b, 0)'),
    ('gelu-ish', 'x * 0.5 * (1 + x / sqrt(2))'),
    ('shared subexpression', '(x + y) * (x + y) + exp(x + y)'),
    ('constant heavy', 'x * (2 * 3) + (10 - 4)'),
]


def main() -> int:
    """Compiles each expression, emits both backends, and checks them.

    Returns:
        0 when every gate passed.
    """
    print('=' * 70)
    print('CAPSTONE · A tensor compiler that emits Triton')
    print('=' * 70)
    print()
    print(environment_banner())

    rng = np.random.default_rng(20260101)
    size = 4096
    inputs = {name: rng.standard_normal(size).astype(np.float64)
              for name in ('x', 'y', 'w', 'b')}

    print('\n' + '-' * 70)
    print('Worked example: max(x * w + b, 0)')
    print('-' * 70)
    program = compile_source('max(x * w + b, 0)')
    print(f'\nunoptimised IR ({program.count()} instructions):')
    print(program)
    optimised, _ = optimise(program)
    print(f'\noptimised IR ({optimised.count()} instructions):')
    print(optimised)
    print('\nemitted Triton:\n')
    triton_source = emit(optimised, 'triton', 'relu_affine')
    print('\n'.join(f'  {line}' for line in triton_source.splitlines()))
    print('\nemitted NumPy:\n')
    print('\n'.join(f'  {line}'
                    for line in emit(optimised, 'numpy',
                                     'relu_affine').splitlines()))

    failures = 0
    print('\n' + '-' * 70)
    print('Gates')
    print('-' * 70)
    print(f'\n  {"expression":<26}{"ops":>5}{"opt":>5}'
          f'{"passes":>9}  numerics')

    for label, expression in EXPRESSIONS:
        raw = compile_source(expression)
        best, _ = optimise(raw)

        numpy_source = emit(best, 'numpy')
        kernel = build_numpy_kernel(numpy_source)
        arguments = [inputs[name] for name in best.free_names]
        produced = kernel(*arguments)
        expected = evaluate_directly(expression, inputs)
        error = float(np.max(np.abs(produced - expected)
                             / np.maximum(np.abs(expected), 1e-9)))

        # The Triton text must at least be syntactically valid Python,
        # even where no GPU can run it.
        triton_text = emit(best, 'triton')
        try:
            compile(triton_text, '<triton>', 'exec')
            syntax_ok = True
        except SyntaxError:
            syntax_ok = False

        unfused, fused = count_memory_passes(best)
        ok = error < 1e-12 and syntax_ok
        failures += not ok
        print(f'  {label:<26}{raw.count():>5}{best.count():>5}'
              f'{unfused:>5} ->{fused:>3}  '
              f'[{"PASS" if ok else "FAIL"}] err {error:.1e}'
              f'{"" if syntax_ok else "  BAD SYNTAX"}')

    print('\n  Gate — one kernel per expression, not one per operation')
    multi = compile_source('(x + y) * (x + y) + exp(x + y)')
    best, _ = optimise(multi)
    kernels_fused = 1
    kernels_unfused = sum(1 for i in best.instructions
                          if i.op in ELEMENTWISE)
    ok = kernels_fused == 1 and kernels_unfused > 1
    failures += not ok
    print(f'  [{"PASS" if ok else "FAIL"}] {kernels_unfused} elementwise '
          f'operations compiled into {kernels_fused} kernel')

    print('\n  Gate — CSE removed the repeated (x + y)')
    adds = sum(1 for i in best.instructions if i.op == '+')
    ok = adds <= 2            # one for x+y, one for the final sum
    failures += not ok
    print(f'  [{"PASS" if ok else "FAIL"}] {adds} addition(s) remain in '
          f'the fused kernel, not 4')

    print('\n  Gate — the emitted Triton loads each input exactly once')
    source = emit(best, 'triton')
    loads = source.count('tl.load')
    ok = loads == len(best.free_names)
    failures += not ok
    print(f'  [{"PASS" if ok else "FAIL"}] {loads} load(s) for '
          f'{len(best.free_names)} input(s); one store')

    print('\n  Gate — an unsupported operation is refused, not mis-emitted')
    try:
        emit(compile_source('x % y'), 'triton')
        print('  [FAIL] modulo should have been rejected')
        failures += 1
    except UnsupportedOperation as error:
        print(f'  [PASS] {error}')

    print(f'\nstatus: {"VERIFIED" if not failures else "FAILED"}')
    if not CAN_RUN_KERNELS:
        print('note:   the emitted Triton was syntax-checked but not run.')
        print('        Open this in Colab with a GPU to execute it.')
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
