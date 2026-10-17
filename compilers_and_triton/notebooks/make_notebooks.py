"""Generates Colab notebooks for the compilers-and-Triton ladder.

Same principle as `notebooks/make_project_notebooks.py` at the top of
this repository: the modules are the source of truth and the notebooks
are built from their text, so editing a module and re-running this
keeps them in step. Nothing is maintained twice.

Two problems have to be solved for a notebook to be self-contained.

1. The modules import each other — `a3_passes` needs `a2_ir`, the
   capstone needs all three. A notebook has no files beside it, so
   each dependency's source is embedded ahead of the module and the
   import statement is removed.
2. The modules call `main()` under a `__main__` guard. In a notebook
   that would run the whole stage the instant the setup cell executes,
   before the reader has read anything, so the guard is stripped and
   `main()` gets a cell of its own.

What each notebook *says* lives in `stage_specs.py`. This file is
machinery only.

Usage:
    python make_notebooks.py
    python make_notebooks.py --slug owner/repo
    python make_notebooks.py --no-badge
"""

import argparse
import json
import pathlib
import re
import sys
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from stage_specs import SPECS                           # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

DEFAULT_SLUG = 'treaper33-hue/ML-AI-assisted-'
NOTEBOOK_PATH = 'compilers_and_triton/notebooks'
BADGE_MARKER = '<!-- colab-badge -->'

SOURCES: Dict[str, str] = {
    'a1_lexer_parser': 'track_a_compilers/a1_lexer_parser.py',
    'a2_ir': 'track_a_compilers/a2_ir.py',
    'a3_passes': 'track_a_compilers/a3_passes.py',
    'a4_schedule': 'track_a_compilers/a4_schedule.py',
    'runtime': 'track_b_triton/runtime.py',
    'b1_vector_add': 'track_b_triton/b1_vector_add.py',
    'b2_softmax': 'track_b_triton/b2_softmax.py',
    'b3_matmul': 'track_b_triton/b3_matmul.py',
    'tensor_compiler': 'capstone/tensor_compiler.py',
}

GUARD = re.compile(r"\nif __name__ == '__main__':\n\s+sys\.exit\(main\(\)\)"
                   r"\s*$")
LOCAL_IMPORT = re.compile(r'^from (?:' + '|'.join(SOURCES) + r') import\b')
PATH_SETUP = re.compile(r'^(sys\.path\.insert\(|ROOT = pathlib\.Path\()')

# The capstone falls back to a stub banner when the Triton runtime is
# not importable. The notebook embeds the real runtime above it, so the
# whole try/except has to go or the stub would shadow the real one.
CAPSTONE_RUNTIME_BLOCK = (
    'try:\n'
    '    from runtime import CAN_RUN_KERNELS, environment_banner\n'
    'except ImportError:                                    '
    '# pragma: no cover\n'
    '    CAN_RUN_KERNELS = False\n'
    '\n'
    '    def environment_banner() -> str:\n'
    '        """Fallback when the Triton runtime module is '
    'unavailable."""\n'
    "        return 'environment: track_b_triton/runtime.py not "
    "importable'\n"
)

GPU_NOTE = (
    '> **This stage needs a GPU to verify its kernel.** In Colab: '
    '**Runtime ->\n> Change runtime type -> T4 GPU**, then re-run the '
    "setup cell. Triton ships\n> with Colab's PyTorch, so nothing needs "
    'installing.\n>\n> Without a GPU the module still runs and still '
    'teaches: the NumPy reference\n> is checked and the kernel is '
    'reported as `SKIP`. **A skip is not a pass.**'
)

HOW_TO_USE = (
    '## How to use this notebook\n\n'
    '1. Run **Setup**. It defines the stage and runs nothing.\n'
    '2. Run the stage and read its output, gates included.\n'
    '3. Work through the sections below, which take the pieces apart '
    'and\n   invite you to break them.\n\n'
    'The gates are declared *before* the run. When one fails the useful '
    'question\nis which is wrong — the code, or the threshold someone '
    'wrote down before the\nproblem was understood.'
)


def read_module(key: str) -> str:
    """Reads one module's source text.

    Args:
        key: A key of `SOURCES`.

    Returns:
        The file's text.
    """
    return (ROOT / SOURCES[key]).read_text(encoding='utf-8')


def strip_for_notebook(text: str) -> str:
    """Removes everything that only makes sense as a file on disk.

    That means the `sys.path` juggling, the imports of sibling modules
    whose source is embedded instead, and the `__main__` guard. The
    guard matters most: left in, the whole stage would run the moment
    the setup cell executed.

    Args:
        text: The module's source.

    Returns:
        Source suitable for a notebook cell.
    """
    text = text.replace(CAPSTONE_RUNTIME_BLOCK, '')

    kept: List[str] = []
    lines = text.split('\n')
    index = 0
    while index < len(lines):
        line = lines[index]
        if PATH_SETUP.match(line):
            index += 1
            continue
        if LOCAL_IMPORT.match(line):
            # A parenthesised import spans lines; consume until the
            # brackets balance, or it would leave a dangling tail.
            depth = line.count('(') - line.count(')')
            index += 1
            while depth > 0 and index < len(lines):
                depth += lines[index].count('(') - lines[index].count(')')
                index += 1
            continue
        kept.append(line)
        index += 1

    return GUARD.sub('\n', '\n'.join(kept)).rstrip() + '\n'


def setup_cell_source(spec: Dict) -> str:
    """Builds the single defining cell for one stage.

    Args:
        spec: A stage specification.

    Returns:
        The combined source text.
    """
    blocks: List[str] = []
    for dependency in spec['needs']:
        blocks.append(f'# ===== embedded: {SOURCES[dependency]} =====\n'
                      + strip_for_notebook(read_module(dependency)))
    blocks.append(f'# ===== {SOURCES[spec["module"]]} =====\n'
                  + strip_for_notebook(read_module(spec['module'])))
    return '\n\n'.join(blocks)


def _as_source(text: str) -> List[str]:
    """Converts text to the newline-terminated form nbformat expects."""
    parts = text.rstrip('\n').split('\n')
    return [part + '\n' for part in parts[:-1]] + [parts[-1]]


def markdown_cell(text: str) -> Dict:
    """Builds a markdown cell."""
    return {'cell_type': 'markdown', 'metadata': {},
            'source': _as_source(text)}


def code_cell(text: str) -> Dict:
    """Builds a code cell."""
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None,
            'outputs': [], 'source': _as_source(text)}


def badge_cell(slug: str, stem: str, branch: str) -> Dict:
    """Builds the Open-in-Colab badge cell.

    The badge is a link, not an embed of anything executable.

    Args:
        slug: owner/repo.
        stem: Notebook file stem.
        branch: Branch name.

    Returns:
        A markdown cell.
    """
    url = (f'https://colab.research.google.com/github/{slug}/blob/'
           f'{branch}/{NOTEBOOK_PATH}/{stem}.ipynb')
    return markdown_cell(
        f'{BADGE_MARKER}\n'
        f'[![Open In Colab](https://colab.research.google.com/assets/'
        f'colab-badge.svg)]({url})')


def build_notebook(spec: Dict, slug: Optional[str], branch: str) -> Dict:
    """Assembles one stage notebook.

    Args:
        spec: A stage specification.
        slug: owner/repo for the badge, or None for no badge.
        branch: Branch the badge points at.

    Returns:
        An nbformat notebook dictionary.
    """
    cells: List[Dict] = []
    if slug:
        cells.append(badge_cell(slug, spec['stem'], branch))

    heading = (f'# {spec["title"]}\n\n'
               f'**{spec["level"]}** · runtime {spec["runtime"]}\n\n'
               f'---\n\n{spec["summary"]}\n\n')
    if spec['gpu']:
        heading += f'{GPU_NOTE}\n\n'
    heading += f'---\n\n{HOW_TO_USE}'
    cells.append(markdown_cell(heading))

    cells.append(markdown_cell('## Setup\n\nDefines everything. Runs '
                               'nothing.'))
    cells.append(code_cell(
        '#@title Setup — run this cell first (collapsed: it is the whole '
        'stage)\n' + setup_cell_source(spec)))

    cells.append(markdown_cell(
        '## Run it\n\nThe stage from start to finish, with its gates at '
        'the end.'))
    cells.append(code_cell('main()'))

    for title, blurb, code in spec['explore']:
        cells.append(markdown_cell(f'## {title}\n\n{blurb}'))
        cells.append(code_cell(code))

    cells.append(markdown_cell(
        f'---\n\n## Where this sits\n\nThe script version is '
        f'`{SOURCES[spec["module"]]}`, and\n'
        f'`compilers_and_triton/run_all.py` runs the whole ladder in '
        f'order.\n`compilers_and_triton/README.md` explains why the '
        f'compiler track and the\nTriton track are one course rather '
        f'than two.'))

    return {
        'cells': cells,
        'metadata': {
            'colab': {'provenance': [], 'toc_visible': True},
            'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
            'language_info': {'name': 'python'},
        },
        'nbformat': 4,
        'nbformat_minor': 0,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Generates every stage notebook.

    Args:
        argv: Optional argument vector.

    Returns:
        0 on success.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--slug', default=DEFAULT_SLUG,
                        help='owner/repo for the Colab badges')
    parser.add_argument('--branch', default='main')
    parser.add_argument('--no-badge', action='store_true',
                        help='omit the badges entirely')
    parser.add_argument('--out-dir', type=pathlib.Path, default=HERE)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    slug = None if args.no_badge else args.slug
    for spec in SPECS:
        notebook = build_notebook(spec, slug, args.branch)
        path = args.out_dir / f'{spec["stem"]}.ipynb'
        path.write_text(json.dumps(notebook, indent=1), encoding='utf-8')
        print(f'{path.name:<34} {len(notebook["cells"]):>2} cells, '
              f'{path.stat().st_size / 1024:>5.0f} KB')

    print(f'\n{len(SPECS)} notebooks'
          + (f', badges -> {slug}' if slug else ', no badges'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
