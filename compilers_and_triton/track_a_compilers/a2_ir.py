"""A2 · FUNDAMENTAL — From a tree to an instruction list.

A tree is good for parsing and bad for optimising. Optimisation wants a
*flat* sequence where every value has a name and every operation has
exactly one result, because then "is this value used again?" and "have I
computed this before?" are questions you can answer by looking, rather
than by walking a tree.

That representation is **three-address code**: `t3 = mul t1 t2`. Each
instruction names its destination and its operands, and every temporary
is assigned exactly once — static single assignment, the property that
makes the passes in A3 straightforward instead of subtle.

Two things to notice:

* `let` disappears. Scoping was a source-language concern; after
  lowering there are only names, and the environment that mapped them
  is gone. Lowering is where language features turn into machine
  concepts.
* The interpreter here is a *specification*. Every optimisation in A3
  must preserve what it computes, and the gates check exactly that.

Run it:
    python a2_ir.py
"""

import dataclasses
import math
import sys
from typing import Callable, Dict, List, Optional, Tuple, Union

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))

from a1_lexer_parser import (Binary, Call, CompileError, Let, Name,  # noqa: E402
                             Node, Number, Unary, parse)

BUILTINS: Dict[str, Tuple[int, Callable]] = {
    'max': (2, max),
    'min': (2, min),
    'exp': (1, math.exp),
    'log': (1, math.log),
    'sqrt': (1, math.sqrt),
    'abs': (1, abs),
}

BINARY_OPS: Dict[str, Callable] = {
    '+': lambda a, b: a + b,
    '-': lambda a, b: a - b,
    '*': lambda a, b: a * b,
    '/': lambda a, b: a / b,
    '%': lambda a, b: a % b,
    '<': lambda a, b: float(a < b),
    '>': lambda a, b: float(a > b),
    '<=': lambda a, b: float(a <= b),
    '>=': lambda a, b: float(a >= b),
    '==': lambda a, b: float(a == b),
    '!=': lambda a, b: float(a != b),
}

# Operations with no side effects, whose results may be folded, shared
# or deleted when unused. Division is included but guarded: folding a
# division by zero at compile time would turn a runtime error into a
# compile-time crash, which is a change in behaviour, not an
# optimisation.
PURE = set(BINARY_OPS) | set(BUILTINS) | {'neg', 'const', 'copy'}


@dataclasses.dataclass(frozen=True)
class Instruction:
    """One three-address instruction.

    Attributes:
        op: 'const', 'copy', 'neg', a binary operator, or a builtin name.
        dest: The temporary this writes, assigned exactly once.
        args: Operand names, or a single literal for 'const'.
    """

    op: str
    dest: str
    args: Tuple[Union[str, float], ...]

    def __str__(self) -> str:
        """Renders the instruction in assembly-like form."""
        if self.op == 'const':
            return f'{self.dest} = {self.args[0]}'
        if self.op == 'copy':
            return f'{self.dest} = {self.args[0]}'
        if self.op == 'neg':
            return f'{self.dest} = neg {self.args[0]}'
        if self.op in BINARY_OPS:
            return f'{self.dest} = {self.args[0]} {self.op} {self.args[1]}'
        return f'{self.dest} = {self.op}({", ".join(map(str, self.args))})'


@dataclasses.dataclass
class Program:
    """A lowered program.

    Attributes:
        instructions: The instruction list, in order.
        result: The temporary holding the final value.
        free_names: Variables the program reads but does not define.
    """

    instructions: List[Instruction]
    result: str
    free_names: Tuple[str, ...] = ()

    def __str__(self) -> str:
        """Renders the whole program."""
        body = '\n'.join(f'  {instruction}'
                         for instruction in self.instructions)
        return f'{body}\n  return {self.result}'

    def count(self) -> int:
        """Returns the instruction count, the headline size metric."""
        return len(self.instructions)


class Lowerer:
    """Walks a tree and emits three-address code.

    Attributes:
        instructions: Emitted so far.
        environment: Source names mapped to temporaries.
        free_names: Names read without being bound.
    """

    def __init__(self) -> None:
        """Starts an empty program."""
        self.instructions: List[Instruction] = []
        self.environment: Dict[str, str] = {}
        self.free_names: List[str] = []
        self._counter = 0

    def fresh(self) -> str:
        """Returns a new, never-reused temporary name."""
        name = f't{self._counter}'
        self._counter += 1
        return name

    def emit(self, op: str, *args: Union[str, float]) -> str:
        """Appends one instruction and returns its destination.

        Args:
            op: The operation.
            *args: Operands.

        Returns:
            The destination temporary.
        """
        dest = self.fresh()
        self.instructions.append(Instruction(op, dest, tuple(args)))
        return dest

    def lower(self, node: Node) -> str:
        """Lowers a subtree, returning the temporary holding its value.

        Args:
            node: The subtree.

        Returns:
            The destination temporary.

        Raises:
            CompileError: On an unknown builtin or wrong arity.
        """
        if isinstance(node, Number):
            return self.emit('const', node.value)

        if isinstance(node, Name):
            if node.identifier in self.environment:
                return self.environment[node.identifier]
            # A free variable: an input to the program.
            if node.identifier not in self.free_names:
                self.free_names.append(node.identifier)
            return node.identifier

        if isinstance(node, Unary):
            return self.emit('neg', self.lower(node.operand))

        if isinstance(node, Binary):
            left = self.lower(node.left)
            right = self.lower(node.right)
            return self.emit(node.operator, left, right)

        if isinstance(node, Call):
            if node.callee not in BUILTINS:
                raise CompileError(f'unknown function {node.callee!r}', 0)
            arity, _ = BUILTINS[node.callee]
            if len(node.arguments) != arity:
                raise CompileError(
                    f'{node.callee} takes {arity} argument(s), '
                    f'got {len(node.arguments)}', 0)
            lowered = [self.lower(a) for a in node.arguments]
            return self.emit(node.callee, *lowered)

        if isinstance(node, Let):
            value = self.lower(node.value)
            # Bind, lower the body, then restore -- lexical scoping,
            # enforced here and nowhere else downstream.
            shadowed = self.environment.get(node.name)
            self.environment[node.name] = value
            result = self.lower(node.body)
            if shadowed is None:
                self.environment.pop(node.name, None)
            else:
                self.environment[node.name] = shadowed
            return result

        raise TypeError(f'cannot lower {type(node).__name__}')


def lower(node: Node) -> Program:
    """Lowers a tree into a program.

    Args:
        node: The root.

    Returns:
        The lowered program.
    """
    lowerer = Lowerer()
    result = lowerer.lower(node)
    return Program(lowerer.instructions, result,
                   tuple(lowerer.free_names))


def compile_source(source: str) -> Program:
    """Parses and lowers in one step.

    Args:
        source: The program text.

    Returns:
        The lowered program.
    """
    return lower(parse(source))


def execute(program: Program,
            inputs: Optional[Dict[str, float]] = None) -> float:
    """Interprets a program.

    This is the specification every optimisation must preserve: A3
    checks each pass by running the program before and after and
    comparing results on random inputs.

    Args:
        program: The program to run.
        inputs: Values for its free variables.

    Returns:
        The value of the result temporary.

    Raises:
        CompileError: If a free variable has no value.
    """
    environment: Dict[str, float] = dict(inputs or {})
    for name in program.free_names:
        if name not in environment:
            raise CompileError(f'no value supplied for {name!r}', 0)

    def read(operand: Union[str, float]) -> float:
        """Resolves an operand to a number."""
        if isinstance(operand, (int, float)):
            return float(operand)
        return environment[operand]

    for instruction in program.instructions:
        if instruction.op == 'const':
            environment[instruction.dest] = float(instruction.args[0])
        elif instruction.op == 'copy':
            environment[instruction.dest] = read(instruction.args[0])
        elif instruction.op == 'neg':
            environment[instruction.dest] = -read(instruction.args[0])
        elif instruction.op in BINARY_OPS:
            environment[instruction.dest] = BINARY_OPS[instruction.op](
                read(instruction.args[0]), read(instruction.args[1]))
        elif instruction.op in BUILTINS:
            _, function = BUILTINS[instruction.op]
            environment[instruction.dest] = float(
                function(*[read(a) for a in instruction.args]))
        else:
            raise CompileError(f'unknown op {instruction.op!r}', 0)
    return float(environment[program.result])


def evaluate_tree(node: Node,
                  inputs: Optional[Dict[str, float]] = None) -> float:
    """Evaluates a tree directly, without lowering.

    An independent second implementation. Agreement between this and
    `execute` is what makes the lowering gate meaningful: two routes to
    the same number, one of them not involving the IR at all.

    Args:
        node: The tree.
        inputs: Values for free variables.

    Returns:
        The value.
    """
    environment = dict(inputs or {})

    def walk(current: Node) -> float:
        """Recursively evaluates one node."""
        if isinstance(current, Number):
            return current.value
        if isinstance(current, Name):
            return environment[current.identifier]
        if isinstance(current, Unary):
            return -walk(current.operand)
        if isinstance(current, Binary):
            return BINARY_OPS[current.operator](walk(current.left),
                                                walk(current.right))
        if isinstance(current, Call):
            _, function = BUILTINS[current.callee]
            return float(function(*[walk(a) for a in current.arguments]))
        if isinstance(current, Let):
            previous = environment.get(current.name)
            environment[current.name] = walk(current.value)
            result = walk(current.body)
            if previous is None:
                environment.pop(current.name, None)
            else:
                environment[current.name] = previous
            return result
        raise TypeError(f'cannot evaluate {type(current).__name__}')

    return walk(node)


SAMPLES: List[Tuple[str, Dict[str, float]]] = [
    ('1 + 2 * 3', {}),
    ('let x = 4 in x * x + x', {}),
    ('max(a * 2, b + 1)', {'a': 3.0, 'b': 10.0}),
    ('let s = a + b in s * s - s', {'a': 2.0, 'b': 5.0}),
    ('exp(0) + sqrt(16) - abs(-3)', {}),
    ('let x = 2 in let x = x + 1 in x * 10', {}),
]


def main() -> int:
    """Demonstrates lowering and checks the gates.

    Returns:
        0 when every gate passed.
    """
    print('=' * 70)
    print('A2 · FUNDAMENTAL — From a tree to an instruction list')
    print('=' * 70)

    source = 'let s = a + b in s * s - s'
    program = compile_source(source)
    print(f'\nsource:\n  {source}')
    print(f'\nthree-address code (free: {", ".join(program.free_names)}):')
    print(program)
    print('\nNote that `let` has vanished: s was bound to t0, and every')
    print('later use is just t2. Scoping was a front-end concern.')

    print('\nGates — the IR computes what the tree computes')
    failures = 0
    for text, inputs in SAMPLES:
        tree = parse(text)
        direct = evaluate_tree(tree, inputs)
        lowered = compile_source(text)
        through_ir = execute(lowered, inputs)
        ok = abs(direct - through_ir) < 1e-12
        failures += not ok
        print(f'  [{"PASS" if ok else "FAIL"}] {text:<38} '
              f'tree={direct:<10.4f} ir={through_ir:<10.4f} '
              f'({lowered.count()} instructions)')

    print('\nGate — every temporary is assigned exactly once (SSA)')
    for text, _ in SAMPLES:
        lowered = compile_source(text)
        destinations = [i.dest for i in lowered.instructions]
        ok = len(destinations) == len(set(destinations))
        failures += not ok
        if not ok:
            print(f'  [FAIL] {text} reassigns a temporary')
    print(f'  [{"PASS" if not failures else "FAIL"}] '
          f'checked {len(SAMPLES)} programs')

    print('\nGate — shadowing resolves to the inner binding')
    shadow = execute(compile_source('let x = 2 in let x = x + 1 in x * 10'))
    ok = abs(shadow - 30.0) < 1e-12
    failures += not ok
    print(f'  [{"PASS" if ok else "FAIL"}] inner x wins: {shadow} (expect 30)')

    print(f'\nstatus: {"VERIFIED" if not failures else "FAILED"}')
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
