"""Generates Colab-ready notebooks from `exercises.py`.

One source of truth: the exercises, their gates and their reference
solutions all live in `exercises.py`, and this script emits notebooks
from them by reading the source with `inspect`. Edit the exercises, run
this, and the notebooks follow.

The generated notebooks are **self-contained**: everything they need is
embedded, so they run in Colab with no clone, no upload and no pip
install beyond what Colab already ships.

Usage:
    python make_notebooks.py
    python make_notebooks.py --out-dir ../notebooks
"""

import argparse
import dataclasses
import inspect
import json
import pathlib
import re
import sys
import textwrap
from typing import Dict, List, Optional, Sequence, Tuple

HERE = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import exercises as ex                      # noqa: E402 - path set above

PART_FILES = {
    1: ('part1_mathematical_engine',
        'Part I — The Mathematical Engine',
        'Norms, Gram–Schmidt, LU, conditioning, eigenpairs, the SVD, PCA, '
        'gradient descent and its stability limit, Jacobians, '
        'backpropagation, entropy, KL divergence and maximum likelihood.'),
    2: ('part2_learning_from_data',
        'Part II — Learning from Data',
        'Cost-based thresholds, the leakage demonstration, leak-safe '
        'scaling, ridge, the logistic gradient, soft-thresholding, Gini '
        'splits, precision and recall, bootstrap intervals, k-means.'),
    3: ('part3_neural_networks',
        'Part III — Neural Networks in PyTorch',
        'Numerically stable softmax, the p − y gradient, convolution by '
        'im2col and its arithmetic, scaled dot-product attention, and a '
        'PyTorch training step.'),
    4: ('part4_sequences_uncertainty',
        'Part IV — Sequences and Uncertainty',
        'Autocorrelation, decomposition, rolling-origin folds, MASE and '
        'its scale invariance, APE at zeros, conformal coverage and the '
        'Winkler interval score.'),
}

HARNESS = '''
# ---- exercise harness -------------------------------------------------
# `check('1.1')` runs that exercise's gate against whatever you have
# defined in this notebook. TODO means the function is missing or still
# raises; it is never a pass.

def _impl_namespace():
    """Exposes the notebook's own functions to the gate functions."""

    class _Namespace:
        def __getattr__(self, name):
            value = globals().get(name)
            if callable(value):
                return value
            raise NotImplementedYet(f'{name} is not defined yet')

    return _Namespace()


_GATES = {}


def check(key):
    """Runs one exercise's gate and prints the verdict."""
    if key not in _GATES:
        raise KeyError(f'no exercise {key!r} in this notebook')
    title, gate, needs_torch = _GATES[key]
    if needs_torch and not _HAS_TORCH:
        print(f'[SKIP] {key}  {title}')
        print('       PyTorch is not available in this runtime.')
        return
    try:
        detail = gate(_impl_namespace())
        print(f'[PASS] {key}  {title}')
        if detail:
            print(f'       {detail}')
    except NotImplementedYet as todo:
        print(f'[TODO] {key}  {title}')
        print(f'       not implemented yet: {todo}')
    except AssertionError as failure:
        print(f'[FAIL] {key}  {title}')
        print(f'       {failure or "gate failed"}')
    except Exception as error:                      # noqa: BLE001
        print(f'[FAIL] {key}  {title}')
        print(f'       {type(error).__name__}: {error}')


def check_all():
    """Runs every gate in this notebook and prints a summary."""
    counts = {'PASS': 0, 'FAIL': 0, 'TODO': 0, 'SKIP': 0}
    for key, (title, gate, needs_torch) in _GATES.items():
        if needs_torch and not _HAS_TORCH:
            counts['SKIP'] += 1
            continue
        try:
            gate(_impl_namespace())
            counts['PASS'] += 1
        except NotImplementedYet:
            counts['TODO'] += 1
        except Exception:                           # noqa: BLE001
            counts['FAIL'] += 1
    print(f'passed {counts["PASS"]}   failed {counts["FAIL"]}   '
          f'todo {counts["TODO"]}   skipped {counts["SKIP"]}')
    if counts['TODO']:
        print(f'\\n{counts["TODO"]} still TODO. A TODO is not a pass.')
'''


def markdown_cell(lines: Sequence[str]) -> Dict:
    """Builds a markdown cell.

    Args:
        lines: Lines of markdown, without trailing newlines.

    Returns:
        An nbformat cell dictionary.
    """
    return {'cell_type': 'markdown', 'metadata': {},
            'source': _as_source(lines)}


def code_cell(lines: Sequence[str]) -> Dict:
    """Builds a code cell.

    Args:
        lines: Lines of Python, without trailing newlines.

    Returns:
        An nbformat cell dictionary.
    """
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None,
            'outputs': [], 'source': _as_source(lines)}


def _as_source(lines: Sequence[str]) -> List[str]:
    """Converts lines to the newline-terminated form nbformat expects."""
    text = '\n'.join(lines).rstrip('\n')
    parts = text.split('\n')
    return [part + '\n' for part in parts[:-1]] + [parts[-1]]


def student_stubs() -> Dict[str, Tuple[str, str]]:
    """Maps each exercise key to its stub's name and source.

    The mapping is read from the `NotImplementedYet('<key> <name>')`
    message inside each stub, so adding an exercise to `exercises.py`
    automatically adds it here.

    Returns:
        A mapping key -> (function name, source text).
    """
    found: Dict[str, Tuple[str, str]] = {}
    for name, member in vars(ex.Student).items():
        if not isinstance(member, staticmethod):
            continue
        source = inspect.getsource(member.__func__)
        match = re.search(r"NotImplementedYet\('([\d.]+)\s+(\w+)'\)", source)
        if not match:
            continue
        # dedent, not cleandoc: cleandoc dedents the first line and the
        # rest by their own common margin, which flattens the body onto
        # the `def` line and produces an IndentationError.
        body = textwrap.dedent(source).rstrip()
        body = '\n'.join(line for line in body.splitlines()
                         if line.strip() != '@staticmethod')
        found[match.group(1)] = (name, body)
    return found


def gate_sources(exercises: Sequence['ex.Exercise']) -> str:
    """Returns the source of every gate function these exercises need.

    Args:
        exercises: The exercises to cover.

    Returns:
        The concatenated source text.
    """
    seen, blocks = set(), []
    for exercise in exercises:
        name = exercise.check.__name__
        if name in seen:
            continue
        seen.add(name)
        blocks.append(inspect.getsource(exercise.check).rstrip())
    return '\n\n\n'.join(blocks)


def setup_cell(exercises: Sequence['ex.Exercise']) -> Dict:
    """Builds the one cell that must be run before anything else.

    It embeds the harness, the gate functions and the reference
    solutions -- the last because two gates compare against them. It is
    written as a Colab form cell so it collapses by default.

    Args:
        exercises: The exercises in this notebook.

    Returns:
        A code cell.
    """
    lines = [
        '#@title Setup — run this cell first '
        '(collapsed: it holds the gates and the reference solutions)',
        '',
        'import math',
        'import time',
        'from typing import Callable, Dict, List, Optional, Sequence, Tuple',
        '',
        'import numpy as np',
        '',
        'try:',
        '    import torch',
        '    _HAS_TORCH = True',
        'except ImportError:',
        '    _HAS_TORCH = False',
        '',
        '',
        'class NotImplementedYet(NotImplementedError):',
        '    """Raised by a stub you have not filled in yet."""',
        '',
        '',
        inspect.getsource(ex._rng).rstrip(),
        '',
        '',
        inspect.getsource(ex._fit_predict).rstrip(),
        '',
        '',
        inspect.getsource(ex._r2).rstrip(),
        '',
        '',
        '# Reference implementations. Two gates compare against these, so',
        '# they have to be here. Reading them before you have tried is a',
        '# way of not learning the material.',
        inspect.getsource(ex.Solutions).rstrip(),
        '',
        '',
        '# ---- gates ----------------------------------------------------',
        gate_sources(exercises),
        '',
        HARNESS.strip(),
        '',
        '',
        '_GATES = {',
    ]
    for exercise in exercises:
        lines.append(f"    {exercise.key!r}: ({exercise.title!r}, "
                     f"{exercise.check.__name__}, {exercise.needs_torch!r}),")
    lines += [
        '}',
        '',
        "print(f'setup complete: {len(_GATES)} exercises loaded"
        ", torch={_HAS_TORCH}')",
    ]
    return code_cell(lines)


def header_cells(part: int, title: str, blurb: str,
                 exercises: Sequence['ex.Exercise']) -> List[Dict]:
    """Builds the title and instructions cells.

    Args:
        part: Part number.
        title: Part title.
        blurb: One-paragraph description.
        exercises: The exercises in this notebook.

    Returns:
        A list of cells.
    """
    rows = [f'| `{e.key}` | {e.title} | {e.anchor} |' for e in exercises]
    return [markdown_cell([
        f'# {title}',
        '',
        '**Exercises from *Machine Learning: The Working Text*.**',
        '',
        blurb,
        '',
        '---',
        '',
        '## How to use this notebook',
        '',
        '1. Run the **Setup** cell below once.',
        '2. Work down the notebook. Each exercise has a stub cell — '
        'replace the',
        '   `raise NotImplementedYet(...)` with your own code and run it.',
        '3. Run the `check(...)` cell under it. It prints `PASS`, `FAIL` '
        'or `TODO`.',
        '4. `check_all()` at the bottom summarises the notebook.',
        '',
        '**A `TODO` is never a pass.** The gate thresholds come from the '
        'book, so',
        'when one fails the useful question is which of the two is wrong — '
        'your',
        'code, or your understanding of the threshold.',
        '',
        'Nothing needs installing: Colab already has NumPy and PyTorch. '
        'Nothing',
        'is downloaded, and no file is read from disk.',
        '',
        '### Exercises in this notebook',
        '',
        '| id | exercise | read first |',
        '|---|---|---|',
        *rows,
    ])]


def exercise_cells(exercise: 'ex.Exercise',
                   stub: Optional[Tuple[str, str]]) -> List[Dict]:
    """Builds the cells for one exercise.

    Args:
        exercise: The exercise.
        stub: Its (name, source), or None when the exercise is an
            observation that needs no implementation.

    Returns:
        A list of cells.
    """
    heading = [
        f'## {exercise.key} · {exercise.title}',
        '',
        f'*Read first: {exercise.anchor}*',
    ]
    if exercise.needs_torch:
        heading += ['', '> Needs PyTorch. Colab has it preinstalled.']
    cells = [markdown_cell(heading)]

    if stub is None:
        cells.append(markdown_cell([
            'Nothing to implement here — this one is an **observation**. '
            'Run the',
            'check and read the number it reports; it is about your '
            'machine or the',
            'mathematics, not about your code.',
        ]))
    else:
        cells.append(code_cell([stub[1]]))
    cells.append(code_cell([f"check('{exercise.key}')"]))
    return cells


def build_notebook(part: int, title: str, blurb: str,
                   exercises: Sequence['ex.Exercise']) -> Dict:
    """Assembles one notebook.

    Args:
        part: Part number.
        title: Part title.
        blurb: Description.
        exercises: Exercises in order.

    Returns:
        An nbformat notebook dictionary.
    """
    stubs = student_stubs()
    cells = header_cells(part, title, blurb, exercises)
    cells.append(setup_cell(exercises))
    for exercise in exercises:
        cells.extend(exercise_cells(exercise, stubs.get(exercise.key)))
    cells.append(markdown_cell([
        '---',
        '',
        '## Summary',
        '',
        'Run this once you have attempted everything above.',
    ]))
    cells.append(code_cell(['check_all()']))
    return {
        'cells': cells,
        'metadata': {
            'colab': {'provenance': [], 'toc_visible': True},
            'kernelspec': {'display_name': 'Python 3',
                           'name': 'python3'},
            'language_info': {'name': 'python'},
        },
        'nbformat': 4,
        'nbformat_minor': 0,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Generates every notebook.

    Args:
        argv: Optional argument vector.

    Returns:
        0 on success.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=pathlib.Path, default=HERE)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for part, (stem, title, blurb) in PART_FILES.items():
        selected = [e for e in ex.EXERCISES if e.part == part]
        notebook = build_notebook(part, title, blurb, selected)
        path = args.out_dir / f'{stem}.ipynb'
        path.write_text(json.dumps(notebook, indent=1), encoding='utf-8')
        total += len(selected)
        print(f'{path.name:<36} {len(selected):>2} exercises, '
              f'{len(notebook["cells"]):>3} cells')
    print(f'\n{total} exercises across {len(PART_FILES)} notebooks')
    return 0


if __name__ == '__main__':
    sys.exit(main())
