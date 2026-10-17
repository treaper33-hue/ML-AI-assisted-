"""Executes the generated notebooks to prove they run.

Two passes per notebook:

  1. **As shipped.** Every cell runs without raising, and every gate
     reports TODO (or SKIP) -- never FAIL, which would mean the stub or
     the harness is broken rather than merely unfilled.
  2. **As completed.** The reference implementations are bound to the
     names the exercises expect, simulating a reader who has finished
     the notebook, and every gate must then report PASS or SKIP.

Without pass 2 a notebook could ship with a gate nobody could ever
satisfy.

Usage:
    python validate_notebooks.py
"""

import contextlib
import io
import json
import pathlib
import sys
from typing import Dict, List, Optional, Sequence, Tuple

HERE = pathlib.Path(__file__).resolve().parent


def code_cells(notebook: Dict) -> List[str]:
    """Returns the source of every code cell, in order.

    Args:
        notebook: A parsed notebook.

    Returns:
        A list of source strings.
    """
    return [''.join(cell['source']) for cell in notebook['cells']
            if cell['cell_type'] == 'code']


def run_notebook(path: pathlib.Path,
                 complete: bool) -> Tuple[List[str], Optional[str]]:
    """Executes a notebook's code cells in one namespace.

    Args:
        path: The notebook file.
        complete: When True, bind the reference solutions to the
            exercise names after setup, simulating a finished notebook.

    Returns:
        A tuple (verdict_lines, error) where error is None on success.
    """
    notebook = json.loads(path.read_text(encoding='utf-8'))
    namespace: Dict[str, object] = {'__name__': '__notebook__'}
    output = io.StringIO()
    cells = code_cells(notebook)

    try:
        for index, source in enumerate(cells):
            with contextlib.redirect_stdout(output):
                exec(compile(source, f'{path.name}#cell{index}', 'exec'),
                     namespace)
    except Exception as error:                      # noqa: BLE001
        return (output.getvalue().splitlines(),
                f'{type(error).__name__}: {error}')

    if not complete:
        return output.getvalue().splitlines(), None

    # Bind every reference implementation to its bare name, simulating a
    # reader who finished the notebook, then re-run the gates. This has
    # to happen AFTER every cell, because each stub cell defines the
    # very name we are replacing.
    solutions = namespace['Solutions']
    for name in dir(solutions):
        if not name.startswith('_'):
            namespace[name] = getattr(solutions, name)

    rerun = io.StringIO()
    try:
        with contextlib.redirect_stdout(rerun):
            for key in list(namespace['_GATES']):
                namespace['check'](key)
    except Exception as error:                      # noqa: BLE001
        return rerun.getvalue().splitlines(), f'{type(error).__name__}: {error}'
    return rerun.getvalue().splitlines(), None


def summarise(lines: Sequence[str]) -> Dict[str, int]:
    """Counts the verdict markers in captured output.

    Args:
        lines: Captured stdout lines.

    Returns:
        A mapping from verdict to count.
    """
    counts = {'PASS': 0, 'FAIL': 0, 'TODO': 0, 'SKIP': 0}
    for line in lines:
        for verdict in counts:
            if line.startswith(f'[{verdict}]'):
                counts[verdict] += 1
    return counts


def main() -> int:
    """Validates every notebook in this directory.

    Returns:
        0 when every notebook passes both checks.
    """
    # part*.ipynb only: the project notebooks have their own
    # validator, and executing their cells here would run whole
    # projects looking for exercise gates that are not there.
    notebooks = sorted(HERE.glob('part*.ipynb'))
    if not notebooks:
        print('no notebooks found; run make_notebooks.py first')
        return 2

    print('=' * 74)
    print('NOTEBOOK VALIDATION')
    print('=' * 74)

    failures = 0
    for path in notebooks:
        print(f'\n{path.name}')

        shipped, error = run_notebook(path, complete=False)
        if error:
            print(f'   as shipped   : CRASHED  {error}')
            failures += 1
            continue
        counts = summarise(shipped)
        bad = counts['FAIL']
        print(f'   as shipped   : {counts["TODO"]} TODO, '
              f'{counts["PASS"]} PASS, {counts["SKIP"]} SKIP, '
              f'{bad} FAIL')
        if bad:
            print('      a stub that FAILS rather than TODOs is broken')
            failures += 1

        completed, error = run_notebook(path, complete=True)
        if error:
            print(f'   as completed : CRASHED  {error}')
            failures += 1
            continue
        counts = summarise(completed)
        unsatisfiable = counts['FAIL'] + counts['TODO']
        print(f'   as completed : {counts["PASS"]} PASS, '
              f'{counts["SKIP"]} SKIP, {counts["FAIL"]} FAIL, '
              f'{counts["TODO"]} TODO')
        if unsatisfiable:
            print('      a gate that cannot be satisfied by the reference '
                  'solution is a bug')
            failures += 1

    print('\n' + '=' * 74)
    if failures:
        print(f'{failures} problem(s) found')
        return 1
    print(f'{len(notebooks)} notebooks execute cleanly, and every gate is '
          f'satisfiable')
    return 0


if __name__ == '__main__':
    sys.exit(main())
