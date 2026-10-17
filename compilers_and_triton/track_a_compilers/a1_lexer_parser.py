"""A1 · FUNDAMENTAL — From text to a tree.

The first thing any compiler does, and the only part most people ever
write by hand. A lexer turns characters into tokens; a parser turns
tokens into a tree that encodes precedence and associativity so that
nothing downstream has to think about either again.

The language is deliberately tiny — arithmetic, comparisons, variables,
`let` bindings and calls — because the *shape* of the pipeline is what
transfers, not the grammar. Every later stage in this track consumes the
tree this module produces.

Why a Pratt parser: precedence climbing expresses operator precedence as
data (a table of binding powers) rather than as a tower of mutually
recursive functions. Adding an operator is a one-line change. This is
the same technique production parsers use.

Run it:
    python a1_lexer_parser.py
"""

import dataclasses
import re
import sys
from typing import Dict, List, Sequence, Tuple, Union

# --------------------------------------------------------------------- #
#                              1. Tokens                                 #
# --------------------------------------------------------------------- #

TOKEN_SPEC: List[Tuple[str, str]] = [
    ('NUMBER', r'\d+\.\d+|\d+'),
    ('NAME', r'[A-Za-z_][A-Za-z_0-9]*'),
    ('OP', r'<=|>=|==|!=|[-+*/%<>()=,]'),
    ('SKIP', r'[ \t]+'),
    ('MISMATCH', r'.'),
]
TOKEN_RE = re.compile('|'.join(f'(?P<{name}>{pattern})'
                               for name, pattern in TOKEN_SPEC))
KEYWORDS = {'let', 'in'}


@dataclasses.dataclass(frozen=True)
class Token:
    """One lexical token.

    Attributes:
        kind: 'NUMBER', 'NAME', 'KEYWORD', 'OP' or 'EOF'.
        text: The matched source text.
        position: Index into the source, for error messages.
    """

    kind: str
    text: str
    position: int

    def __repr__(self) -> str:
        """Returns a compact form for test output."""
        return f'{self.kind}({self.text!r})'


class CompileError(Exception):
    """Raised for any error in the front end, with a source position."""

    def __init__(self, message: str, position: int, source: str = '') -> None:
        """Builds an error that can point at the offending character.

        Args:
            message: What went wrong.
            position: Index into the source.
            source: The source text, for the caret line.
        """
        self.position = position
        if source:
            caret = ' ' * position + '^'
            message = f'{message}\n  {source}\n  {caret}'
        super().__init__(message)


def tokenize(source: str) -> List[Token]:
    """Converts source text into a token list.

    Args:
        source: The program text.

    Returns:
        Tokens, ending with a single EOF token.

    Raises:
        CompileError: On any character the language does not define.
    """
    tokens: List[Token] = []
    for match in TOKEN_RE.finditer(source):
        kind = match.lastgroup
        text = match.group()
        if kind == 'SKIP':
            continue
        if kind == 'MISMATCH':
            raise CompileError(f'unexpected character {text!r}',
                               match.start(), source)
        if kind == 'NAME' and text in KEYWORDS:
            kind = 'KEYWORD'
        tokens.append(Token(kind, text, match.start()))
    tokens.append(Token('EOF', '', len(source)))
    return tokens


# --------------------------------------------------------------------- #
#                            2. The tree                                 #
# --------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class Number:
    """A literal."""

    value: float


@dataclasses.dataclass(frozen=True)
class Name:
    """A variable reference."""

    identifier: str


@dataclasses.dataclass(frozen=True)
class Unary:
    """Prefix negation."""

    operator: str
    operand: 'Node'


@dataclasses.dataclass(frozen=True)
class Binary:
    """An infix operation."""

    operator: str
    left: 'Node'
    right: 'Node'


@dataclasses.dataclass(frozen=True)
class Call:
    """A function call."""

    callee: str
    arguments: Tuple['Node', ...]


@dataclasses.dataclass(frozen=True)
class Let:
    """A binding: `let name = value in body`."""

    name: str
    value: 'Node'
    body: 'Node'


Node = Union[Number, Name, Unary, Binary, Call, Let]

# Binding power drives precedence. Higher binds tighter. Expressing it
# as data is the whole point: adding an operator is one line here, not a
# new level of recursive descent.
BINDING_POWER: Dict[str, int] = {
    '==': 10, '!=': 10, '<': 10, '>': 10, '<=': 10, '>=': 10,
    '+': 20, '-': 20,
    '*': 30, '/': 30, '%': 30,
}
RIGHT_ASSOCIATIVE: set = set()


class Parser:
    """A Pratt (precedence-climbing) parser.

    Attributes:
        tokens: The token stream.
        source: Original text, for error messages.
        index: Current position in the stream.
    """

    def __init__(self, tokens: Sequence[Token], source: str = '') -> None:
        """Stores the stream.

        Args:
            tokens: Tokens from `tokenize`.
            source: The original text.
        """
        self.tokens = list(tokens)
        self.source = source
        self.index = 0

    @property
    def current(self) -> Token:
        """The token under the cursor."""
        return self.tokens[self.index]

    def advance(self) -> Token:
        """Consumes and returns the current token."""
        token = self.current
        self.index += 1
        return token

    def expect(self, text: str) -> Token:
        """Consumes a token that must have the given text.

        Args:
            text: The expected token text.

        Returns:
            The consumed token.

        Raises:
            CompileError: If the next token does not match.
        """
        if self.current.text != text:
            raise CompileError(
                f'expected {text!r}, found {self.current.text!r}',
                self.current.position, self.source)
        return self.advance()

    def parse(self) -> Node:
        """Parses one complete expression.

        Returns:
            The root of the tree.

        Raises:
            CompileError: If tokens remain after the expression.
        """
        node = self.expression(0)
        if self.current.kind != 'EOF':
            raise CompileError(f'unexpected {self.current.text!r}',
                               self.current.position, self.source)
        return node

    def expression(self, minimum_power: int) -> Node:
        """Parses an expression, stopping below a binding power.

        This single loop is the whole precedence mechanism: keep
        absorbing operators while they bind at least as tightly as the
        caller allows, and recurse with a higher floor for the
        right-hand side.

        Args:
            minimum_power: The caller's binding power floor.

        Returns:
            The parsed subtree.
        """
        left = self.prefix()
        while True:
            token = self.current
            power = BINDING_POWER.get(token.text, 0)
            if token.kind != 'OP' or power < minimum_power or power == 0:
                break
            self.advance()
            next_floor = (power if token.text in RIGHT_ASSOCIATIVE
                          else power + 1)
            right = self.expression(next_floor)
            left = Binary(token.text, left, right)
        return left

    def prefix(self) -> Node:
        """Parses a prefix position: literal, name, call, group or `let`.

        Returns:
            The parsed subtree.

        Raises:
            CompileError: On anything that cannot begin an expression.
        """
        token = self.advance()

        if token.kind == 'NUMBER':
            return Number(float(token.text))

        if token.kind == 'KEYWORD' and token.text == 'let':
            name = self.advance()
            if name.kind != 'NAME':
                raise CompileError('expected a name after `let`',
                                   name.position, self.source)
            self.expect('=')
            value = self.expression(0)
            keyword = self.advance()
            if keyword.text != 'in':
                raise CompileError('expected `in` after the bound value',
                                   keyword.position, self.source)
            return Let(name.text, value, self.expression(0))

        if token.kind == 'NAME':
            if self.current.text == '(':
                self.advance()
                arguments: List[Node] = []
                if self.current.text != ')':
                    arguments.append(self.expression(0))
                    while self.current.text == ',':
                        self.advance()
                        arguments.append(self.expression(0))
                self.expect(')')
                return Call(token.text, tuple(arguments))
            return Name(token.text)

        if token.text == '(':
            inner = self.expression(0)
            self.expect(')')
            return inner

        if token.text == '-':
            return Unary('-', self.expression(max(BINDING_POWER.values()) + 1))

        raise CompileError(f'{token.text!r} cannot start an expression',
                           token.position, self.source)


def parse(source: str) -> Node:
    """Parses source text into a tree.

    Args:
        source: The program.

    Returns:
        The root node.
    """
    return Parser(tokenize(source), source).parse()


def render(node: Node) -> str:
    """Renders a tree back to fully parenthesised source.

    Round-tripping through this is how the gates below check that
    precedence was parsed the way the language claims.

    Args:
        node: Any node.

    Returns:
        A parenthesised string.
    """
    if isinstance(node, Number):
        value = node.value
        return str(int(value)) if value == int(value) else str(value)
    if isinstance(node, Name):
        return node.identifier
    if isinstance(node, Unary):
        return f'(-{render(node.operand)})'
    if isinstance(node, Binary):
        return f'({render(node.left)} {node.operator} {render(node.right)})'
    if isinstance(node, Call):
        inner = ', '.join(render(a) for a in node.arguments)
        return f'{node.callee}({inner})'
    if isinstance(node, Let):
        return (f'(let {node.name} = {render(node.value)} '
                f'in {render(node.body)})')
    raise TypeError(f'unknown node {type(node).__name__}')


GATES: List[Tuple[str, str, str]] = [
    ('precedence', '1 + 2 * 3', '(1 + (2 * 3))'),
    ('left associativity', '1 - 2 - 3', '((1 - 2) - 3)'),
    ('parentheses override', '(1 + 2) * 3', '((1 + 2) * 3)'),
    ('unary binds tightest', '-2 * 3', '((-2) * 3)'),
    ('comparison is loosest', 'a + 1 < b * 2',
     '((a + 1) < (b * 2))'),
    ('calls', 'max(a, b + 1)', 'max(a, (b + 1))'),
    ('let', 'let x = 2 in x * x', '(let x = 2 in (x * x))'),
    ('nested let', 'let x = 1 in let y = 2 in x + y',
     '(let x = 1 in (let y = 2 in (x + y)))'),
]


def main() -> int:
    """Demonstrates the front end and checks the gates.

    Returns:
        0 when every gate passed.
    """
    print('=' * 70)
    print('A1 · FUNDAMENTAL — From text to a tree')
    print('=' * 70)

    sample = 'let scale = 2 in max(scale * x + 1, y - 3)'
    print(f'\nsource : {sample}')
    print('tokens :', ' '.join(repr(t) for t in tokenize(sample)[:8]), '...')
    print(f'tree   : {render(parse(sample))}')

    print('\nGates — precedence and associativity, checked by round-trip')
    failures = 0
    for name, source, expected in GATES:
        actual = render(parse(source))
        ok = actual == expected
        failures += not ok
        print(f'  [{"PASS" if ok else "FAIL"}] {name:<24} '
              f'{source:<34} -> {actual}')
        if not ok:
            print(f'{"":9}expected {expected}')

    print('\nErrors point at the character')
    for bad in ('1 + * 2', 'let = 3 in 1', '2 $ 3'):
        try:
            parse(bad)
            print(f'  [FAIL] {bad!r} should not have parsed')
            failures += 1
        except CompileError as error:
            first = str(error).splitlines()[0]
            print(f'  [PASS] {bad!r:<16} {first}')

    print(f'\nstatus: {"VERIFIED" if not failures else "FAILED"}')
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
