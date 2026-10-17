"""Executes the project notebooks far enough to prove they work.

Running all five projects end to end takes a couple of minutes and
`projects/run_all.py` already does exactly that. What this checks is the
part that is specific to the *notebook* form, and that is where the
breakage actually lives:

  * every code cell compiles;
  * the Setup cell executes and defines `main` without running anything
    (a notebook that trains the moment you open it is a bad notebook);
  * `__file__` is handled, since a notebook cell has none and the
    projects use it to locate `runs/`;
  * one project is run end to end inside a notebook namespace, to prove
    the pattern holds and not merely that the text parses.

Usage:
    python validate_project_notebooks.py
    python validate_project_notebooks.py --run-all   every project
"""

import argparse
import contextlib
import io
import json
import pathlib
import sys
import time
import types
from typing import Dict, List, Optional, Sequence, Tuple

HERE = pathlib.Path(__file__).resolve().parent

# Project 3 runs in about two seconds, so it is the one executed end to
# end on every validation.
QUICK_RUN = 'project3_forecast_conformal.ipynb'


def code_cells(notebook: Dict) -> List[str]:
    """Returns the source of every code cell, in order."""
    return [''.join(cell['source']) for cell in notebook['cells']
            if cell['cell_type'] == 'code']


def check_notebook(path: pathlib.Path,
                   run_project: bool) -> Tuple[List[str], bool]:
    """Validates one notebook.

    Args:
        path: The notebook file.
        run_project: Whether to execute the project's `main`.

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

    # Execute inside a real module registered in sys.modules, which is
    # what Jupyter and Colab do. A bare dict is not faithful:
    # `dataclasses` resolves string annotations via
    # `sys.modules[cls.__module__].__dict__`, so a class whose module is
    # not registered raises an AttributeError that no real notebook
    # would ever hit.
    module_name = f'_nb_{path.stem}'
    module = types.ModuleType(module_name)
    namespace = module.__dict__
    sys.modules[module_name] = module
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec(compile(cells[0], f'{path.name}#setup', 'exec'), namespace)
    except Exception as error:                      # noqa: BLE001
        messages.append(f'setup cell raised: {type(error).__name__}: {error}')
        return messages, False
    finally:
        sys.modules.pop(module_name, None)

    if not callable(namespace.get('main')):
        messages.append('setup cell did not define main()')
        return messages, False
    if '__file__' not in namespace:
        messages.append('__file__ is not defined; the project cannot '
                        'locate runs/')
        ok = False
    noise = output.getvalue().strip()
    if noise:
        messages.append(f'setup printed {len(noise.splitlines())} line(s) '
                        f'-- it should define, not run')
        ok = False
    else:
        messages.append('setup defines everything and runs nothing')

    if run_project:
        started = time.perf_counter()
        run_output = io.StringIO()
        try:
            with contextlib.redirect_stdout(run_output):
                code = namespace['main']([])
        except Exception as error:                  # noqa: BLE001
            messages.append(f'main() raised: {type(error).__name__}: {error}')
            return messages, False
        elapsed = time.perf_counter() - started
        text = run_output.getvalue()
        verified = 'status: VERIFIED' in text
        messages.append(f'main() returned {code} in {elapsed:.1f}s, '
                        f'{"VERIFIED" if verified else "NOT verified"}')
        ok = ok and code == 0 and verified

    return messages, ok


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Validates every project notebook.

    Args:
        argv: Optional argument vector.

    Returns:
        0 when all of them pass.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-all', action='store_true',
                        help='execute every project, not just the quick one')
    args = parser.parse_args(argv)

    notebooks = sorted(HERE.glob('project*.ipynb'))
    if not notebooks:
        print('no project notebooks; run make_project_notebooks.py first')
        return 2

    print('=' * 74)
    print('PROJECT NOTEBOOK VALIDATION')
    print('=' * 74)

    failures = 0
    for path in notebooks:
        run_it = args.run_all or path.name == QUICK_RUN
        print(f'\n{path.name}')
        messages, ok = check_notebook(path, run_it)
        for message in messages:
            print(f'   {message}')
        if not ok:
            failures += 1
            print('   ^ PROBLEM')

    print('\n' + '=' * 74)
    if failures:
        print(f'{failures} notebook(s) have problems')
        return 1
    print(f'{len(notebooks)} project notebooks are sound')
    return 0


if __name__ == '__main__':
    sys.exit(main())
