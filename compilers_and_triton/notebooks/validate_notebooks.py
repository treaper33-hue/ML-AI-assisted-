"""Executes every cell of every stage notebook, and says so honestly.

A notebook that has not been executed is a notebook that does not
work. Generated notebooks fail in ways the source files never do —
imports that were stripped and were needed, a name that only existed
because of `sys.path`, a cell written against an older signature — so
this runs them rather than parsing them.

What it checks, per notebook:

  * every code cell compiles;
  * the Setup cell executes and **prints nothing**, because a notebook
    that runs the whole stage the moment you open it is a bad notebook;
  * `main` exists afterwards;
  * the run cell executes and reports VERIFIED, or UNVERIFIED where a
    GPU is needed and absent;
  * **every exploration cell executes**, in order, in the same
    namespace, since those cells exist only in the notebook and are
    not covered by anything else in this repository.

The namespace is a real module registered in `sys.modules`, which is
what Jupyter and Colab do. A bare dict is not faithful: `dataclasses`
resolves string annotations through `sys.modules[cls.__module__]`, and
these modules are full of dataclasses.

Usage:
    python validate_notebooks.py
    python validate_notebooks.py --only a3
"""

import argparse
import contextlib
import io
import json
import linecache
import pathlib
import sys
import time
import types
from typing import Dict, List, Optional, Sequence, Tuple

HERE = pathlib.Path(__file__).resolve().parent

ACCEPTABLE = ('status: VERIFIED', 'status: UNVERIFIED')


def code_cells(notebook: Dict) -> List[str]:
    """Returns the source of every code cell, in order."""
    return [''.join(cell['source']) for cell in notebook['cells']
            if cell['cell_type'] == 'code']


def check_notebook(path: pathlib.Path) -> Tuple[List[str], bool]:
    """Runs one notebook from top to bottom.

    Args:
        path: The notebook file.

    Returns:
        A tuple (messages, ok).
    """
    notebook = json.loads(path.read_text(encoding='utf-8'))
    cells = code_cells(notebook)
    messages: List[str] = []
    ok = True

    for index, source in enumerate(cells):
        try:
            compile(source, f'{path.name}#cell{index}', 'exec')
        except SyntaxError as error:
            messages.append(f'cell {index} does not compile: {error}')
            return messages, False
    messages.append(f'{len(cells)} cells compile')

    module_name = f'_nb_{path.stem}'
    module = types.ModuleType(module_name)
    namespace = module.__dict__
    sys.modules[module_name] = module

    try:
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output):
                exec_cell(cells[0], namespace, path.name, 0)
        except Exception as error:                      # noqa: BLE001
            messages.append(f'setup raised {type(error).__name__}: {error}')
            return messages, False

        noise = output.getvalue().strip()
        if noise:
            messages.append(f'setup printed {len(noise.splitlines())} '
                            f'line(s) — it should define, not run')
            ok = False
        else:
            messages.append('setup defines everything and runs nothing')

        if not callable(namespace.get('main')):
            messages.append('setup did not define main()')
            return messages, False

        started = time.perf_counter()
        run_output = io.StringIO()
        try:
            with contextlib.redirect_stdout(run_output):
                code = exec_cell(cells[1], namespace, path.name, 1)
        except Exception as error:                      # noqa: BLE001
            messages.append(f'the run cell raised '
                            f'{type(error).__name__}: {error}')
            return messages, False
        del code
        text = run_output.getvalue()
        verdict = next((line.strip() for line in text.splitlines()
                        if line.strip().startswith('status:')),
                       'no status line')
        if not any(marker in text for marker in ACCEPTABLE):
            messages.append(f'run cell reported "{verdict}"')
            ok = False
        else:
            messages.append(f'run cell: {verdict} '
                            f'({time.perf_counter() - started:.1f}s)')

        for index, source in enumerate(cells[2:], start=2):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exec_cell(source, namespace, path.name, index)
            except Exception as error:                  # noqa: BLE001
                messages.append(f'exploration cell {index - 1} raised '
                                f'{type(error).__name__}: {error}')
                ok = False
        messages.append(f'{len(cells) - 2} exploration cells execute')
    finally:
        sys.modules.pop(module_name, None)

    return messages, ok


def exec_cell(source: str, namespace: Dict, name: str, index: int):
    """Executes one cell's source in a namespace, as IPython would.

    The `linecache` registration is not decoration. `@triton.jit` reads
    its function's source with `inspect.getsourcelines`, which consults
    `linecache`, and raises "@jit functions should be defined in a
    Python file" when it cannot. Cell code lives in no file, so IPython
    registers each cell under its pseudo-filename and that is why
    Triton works in Colab at all.

    A harness that skips this step fails on all three GPU notebooks
    while the notebooks themselves are fine -- which is what happened
    on the first GPU run here. Faithfulness to the real runtime is the
    whole value of a validator.

    Args:
        source: The cell's code.
        namespace: The notebook namespace.
        name: Notebook file name, for tracebacks.
        index: Cell index, for tracebacks.

    Returns:
        None; cells are run for their effects.
    """
    filename = f'{name}#cell{index}'
    lines = source.splitlines(keepends=True)
    linecache.cache[filename] = (len(source), None, lines, filename)
    exec(compile(source, filename, 'exec'), namespace)     # noqa: S102
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Validates every stage notebook.

    Args:
        argv: Optional argument vector.

    Returns:
        0 when all of them pass.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', help='substring of the file name')
    args = parser.parse_args(argv)

    notebooks = sorted(p for p in HERE.glob('*.ipynb')
                       if not args.only or args.only in p.name)
    if not notebooks:
        print('no notebooks; run make_notebooks.py first')
        return 2

    print('=' * 74)
    print('COMPILERS AND TRITON — notebook validation')
    print('=' * 74)

    failures = 0
    for path in notebooks:
        print(f'\n{path.name}')
        messages, ok = check_notebook(path)
        for message in messages:
            print(f'   {message}')
        if not ok:
            failures += 1
            print('   ^ PROBLEM')

    print('\n' + '=' * 74)
    if failures:
        print(f'{failures} of {len(notebooks)} notebooks have problems')
        return 1
    print(f'{len(notebooks)} notebooks execute end to end')
    return 0


if __name__ == '__main__':
    sys.exit(main())
