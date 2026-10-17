"""Generates Colab notebooks for the five worked projects.

Same principle as `make_notebooks.py`: the project scripts in
`../projects/` are the source of truth, and the notebooks are built from
their text. Editing a project and re-running this keeps them in step.

Each notebook is self-contained. Project 5 imports from Project 1 when
run as a script; here Project 1's source is embedded ahead of it, so the
notebook needs no files at all.

Usage:
    python make_project_notebooks.py
"""

import argparse
import json
import pathlib
import re
import sys
from typing import Dict, List, Optional, Sequence

HERE = pathlib.Path(__file__).resolve().parent
PROJECTS_DIR = HERE.parent / 'projects'

GUARD = re.compile(r"\nif __name__ == '__main__':\n\s+sys\.exit\(main\(\)\)\s*$")

# Per project: file, title, chapters, runtime, and the cells that let a
# reader poke at the pieces rather than only run the whole thing.
SPECS: List[Dict] = [
    {
        'file': 'p1_tabular_churn.py',
        'stem': 'project1_tabular_churn',
        'title': 'Project 1 — Tabular classification, end to end',
        'chapters': '9–14',
        'runtime': '~40 s with --quick',
        'run_args': "['--quick']",
        'explore': [
            ('The data, and why it is hard',
             'The generator produces the pathologies real data has. Look '
             'at the\nschema card before anything else: an 18% positive '
             'rate, missing-not-at-\nrandom values, and **several rows per '
             'customer**, which is what forces\na group-aware split.',
             "frame, labels, groups = generate_churn_data(seed=20260101)\n"
             "schema = describe_schema(frame, labels, groups)\n"
             "for key, value in schema.items():\n"
             "    print(f'{key:<18} {value}')\n"
             "frame.head()"),
            ('The threshold follows from the costs',
             'Chapter 9 derived p* = C_FP / (C_FP + C_FN). Change the two '
             'costs and\nwatch the optimum move. The default 0.5 is the '
             'special case where both\nerrors cost the same — which is '
             'almost never true.',
             "for fp, fn in [(5, 120), (1, 1), (50, 60), (1, 1000)]:\n"
             "    print(f'C_fp={fp:<4} C_fn={fn:<5} -> '\n"
             "          f'optimal threshold {fp / (fp + fn):.4f}')"),
            ('Try it yourself',
             'Widen the search grid and watch the optimism gap grow — with '
             'four\nconfigurations there is almost nothing to overfit, '
             'which is why the\n`--quick` run shows a gap near zero.',
             "main([])   # the full search; slower, and the gap is larger"),
        ],
    },
    {
        'file': 'p2_digits_cnn.py',
        'stem': 'project2_digits_cnn',
        'title': 'Project 2 — A convolutional classifier, trained by hand',
        'chapters': '16–19',
        'runtime': '~10 s on CPU',
        'run_args': "['--epochs', '15']",
        'explore': [
            ('The gradient check, on its own',
             'Run this before trusting any training. Double precision and '
             'dropout\noff: float32 round-off sits above the discrepancy '
             'you are hunting.',
             "print(f'worst relative gap: {gradient_check(20260101):.3e}')"),
            ('Where the learning rate diverges',
             'The loss-versus-rate curve is the 2/L stability edge of '
             'Chapter 6,\nmeasured rather than assumed.',
             "x_train, y_train, *_ = load_data(20260101)\n"
             "lr, curve = lr_range_test(x_train, y_train, 20260101)\n"
             "for rate, loss in curve[::6]:\n"
             "    print(f'lr {rate:8.5f}   smoothed loss {loss:8.4f}')\n"
             "print(f'\\nsuggested: {lr:.4f}')"),
            ('Try it yourself',
             'Push the learning rate past the edge and watch the run fail. '
             'A model\nthat cannot diverge cannot teach you where the limit '
             'is.',
             "main(['--epochs', '10', '--lr', '2.0'])"),
        ],
    },
    {
        'file': 'p3_forecast_conformal.py',
        'stem': 'project3_forecast_conformal',
        'title': 'Project 3 — Forecasting with honest evaluation and real intervals',
        'chapters': '22–25',
        'runtime': '~2 s',
        'run_args': '[]',
        'explore': [
            ('Diagnostics first',
             'ADF and KPSS have opposite nulls, so all four combinations '
             'occur.\nBoth outcomes are recorded; picking whichever agrees '
             'with your plan is\nthe quiet dishonesty nothing in the output '
             'would otherwise reveal.',
             "series, shift_index = generate_series(seed=20260101)\n"
             "print(stationarity_report(series))\n"
             "print('\\nACF:', [round(v, 3) for v in acf(series, 8)])"),
            ('Coverage, with and without the shift',
             'The headline result. Conformal coverage holds while the '
             'process is\nexchangeable and fails when it is not — so run '
             'the stationary control\nand compare.',
             "print('=== with a regime shift ===')\n"
             "main([])\n"
             "print('\\n\\n=== stationary control ===')\n"
             "main(['--no-shift'])"),
            ('Try it yourself',
             'Lengthen the horizon and watch the error curve climb. Beyond '
             'some\nhorizon the model knows nothing the long-run mean does '
             'not.',
             "main(['--horizon', '56', '--folds', '3'])"),
        ],
    },
    {
        'file': 'p4_vae_elbo.py',
        'stem': 'project4_vae_elbo',
        'title': 'Project 4 — A variational autoencoder, with the ELBO taken apart',
        'chapters': '5, 8, 20',
        'runtime': '~20 s on CPU',
        'run_args': "['--epochs', '30']",
        'explore': [
            ('Reproduce the collapse deliberately',
             'With summed squared error over 64 pixels in [0,1], '
             'reconstruction is of\norder 5 while the KL wants tens of '
             'nats — so the cheapest move is to\nswitch the latent off. '
             'That is posterior collapse caused by a units\nmismatch, not '
             'by anything deep.',
             "import torch\n"
             "x = torch.rand(32, 64)\n"
             "mu, logvar = torch.zeros(32, 8), torch.zeros(32, 8)\n"
             "for kind in ('gaussian', 'bernoulli'):\n"
             "    recon, kl = elbo_terms(torch.rand(32, 64), x, mu, logvar,\n"
             "                           kind)\n"
             "    print(f'{kind:<10} reconstruction {float(recon):8.2f}   '\n"
             "          f'KL {float(kl):6.2f}')\n"
             "print('\\nagainst a KL in the tens, a reconstruction of ~5 '\n"
             "      'loses every time')"),
            ('The beta sweep',
             'Five values of beta, five completely different models. The '
             'summed loss\nlooks unremarkable throughout; the split tells '
             'you which one you\nactually trained.',
             "main(['--epochs', '20'])"),
            ('Try it yourself',
             'Shrink the latent and the VAE has less to give up, so the '
             'gap against\nPCA narrows. Try 2, 4 and 16.',
             "main(['--latent', '4', '--epochs', '20', '--skip-sweep'])"),
        ],
    },
    {
        'file': 'p5_serving_monitoring.py',
        'stem': 'project5_serving_monitoring',
        'title': 'Project 5 — Serving a model, and watching it afterwards',
        'chapters': '28',
        'runtime': '~9 s',
        'run_args': "['--requests', '150']",
        'embed': ['p1_tabular_churn.py'],
        'explore': [
            ('Drift detection on its own',
             'PSI bins the live sample by the training data\'s quantiles. '
             'Below 0.1 is\nnothing, above 0.25 is significant. Note it '
             'stays quiet on\nin-distribution traffic — a detector that '
             'fires constantly is worse than\nnone.',
             "frame, labels, groups = generate_churn_data(seed=20260101)\n"
             "cut = int(len(frame) * 0.8)\n"
             "reference, live = frame.iloc[:cut], frame.iloc[cut:]\n"
             "print('in-distribution:')\n"
             "for row in drift_report(reference, live):\n"
             "    print('  ', row)\n"
             "print('\\nafter a covariate shift:')\n"
             "for row in drift_report(reference,\n"
             "                        apply_covariate_shift(live, 1)):\n"
             "    print('  ', row)"),
            ('What the service rejects',
             'Every malformed request must return 400 with a readable '
             'reason — never\na 500, and never a silently imputed '
             'prediction.',
             "good = {f: (1.0 if f in NUMERIC else 'monthly')\n"
             "        for f in NUMERIC + CATEGORICAL}\n"
             "cases = {\n"
             "    'valid': good,\n"
             "    'missing field': {k: v for k, v in good.items()\n"
             "                      if k != 'tenure_months'},\n"
             "    'wrong type': {**good, 'monthly_charge': 'free'},\n"
             "    'infinite': {**good, 'days_since_login': float('inf')},\n"
             "    'not an object': [1, 2, 3],\n"
             "}\n"
             "import p5_shim  # noqa - placeholder removed below"),
            ('Try it yourself',
             'Raise the request count and watch the latency percentiles. '
             'Most of the\n24 ms is pandas and framework overhead, not the '
             'model — batching moves\nit by an order of magnitude, '
             'optimising the model moves it by nothing.',
             "main(['--requests', '400'])"),
        ],
    },
]

# The validation cell above needs the real validator, not a placeholder.
SPECS[4]['explore'][1] = (
    SPECS[4]['explore'][1][0],
    SPECS[4]['explore'][1][1],
    "good = {f: (1.0 if f in NUMERIC else 'monthly')\n"
    "        for f in NUMERIC + CATEGORICAL}\n"
    "cases = {\n"
    "    'valid': good,\n"
    "    'missing field': {k: v for k, v in good.items()\n"
    "                      if k != 'tenure_months'},\n"
    "    'wrong type': {**good, 'monthly_charge': 'free'},\n"
    "    'infinite': {**good, 'days_since_login': float('inf')},\n"
    "    'not an object': [1, 2, 3],\n"
    "}\n"
    "REQUIRED_FIELDS = NUMERIC + CATEGORICAL\n"
    "for label, payload in cases.items():\n"
    "    ok, message = validate_payload(payload)\n"
    "    print(f'{label:<16} {\"accept\" if ok else \"REJECT\":<7} {message}')"
)


def _as_source(text: str) -> List[str]:
    """Converts text to the newline-terminated form nbformat expects."""
    parts = text.rstrip('\n').split('\n')
    return [part + '\n' for part in parts[:-1]] + [parts[-1]]


def markdown_cell(text: str) -> Dict:
    """Builds a markdown cell from a block of text."""
    return {'cell_type': 'markdown', 'metadata': {},
            'source': _as_source(text)}


def code_cell(text: str) -> Dict:
    """Builds a code cell from a block of text."""
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None,
            'outputs': [], 'source': _as_source(text)}


def project_source(name: str) -> str:
    """Reads a project and strips its `__main__` guard.

    Without stripping the guard the whole project would run the moment
    the cell is executed, before the reader has read anything.

    Args:
        name: File name inside `projects/`.

    Returns:
        The source text, guard removed.
    """
    text = (PROJECTS_DIR / name).read_text(encoding='utf-8')
    return GUARD.sub('\n', text).rstrip() + '\n'


def prepare_sources(spec: Dict) -> str:
    """Builds the single defining cell for a project.

    Args:
        spec: The project specification.

    Returns:
        The combined source text.
    """
    # The projects locate their `runs/` directory from `__file__`, which
    # a notebook cell does not have. Defining it against the working
    # directory makes the audit packs land in ./runs, both in Colab and
    # locally.
    blocks = ['import pathlib as _pathlib\n'
              "__file__ = str(_pathlib.Path.cwd() / 'notebook.py')"]
    for embedded in spec.get('embed', []):
        blocks.append(f'# ===== embedded: {embedded} '
                      f'(Project 5 builds on Project 1) =====\n'
                      + project_source(embedded))
    main_source = project_source(spec['file'])
    if spec.get('embed'):
        # The script imports these from Project 1; in the notebook they
        # are already defined above, so the import must not run.
        main_source = main_source.replace(
            'try:\n'
            '    from p1_tabular_churn import CATEGORICAL, NUMERIC\n'
            '    from p1_tabular_churn import build_preprocessor, '
            'generate_churn_data\n'
            '    HAS_P1 = True\n'
            'except ImportError:                 '
            '# pragma: no cover - environment gate\n'
            '    HAS_P1 = False',
            '# Project 1 is embedded above, so there is nothing to import.\n'
            'HAS_P1 = True')
    blocks.append(f'# ===== {spec["file"]} =====\n' + main_source)
    return '\n\n'.join(blocks)


def build_notebook(spec: Dict) -> Dict:
    """Assembles one project notebook.

    Args:
        spec: The project specification.

    Returns:
        An nbformat notebook dictionary.
    """
    docstring = project_source(spec['file']).split('"""')[1].strip()
    summary = '\n'.join(docstring.splitlines()[2:]).strip()

    cells = [markdown_cell(
        f'# {spec["title"]}\n\n'
        f'**From *Machine Learning: The Working Text*, Part VI.**\n\n'
        f'Chapters {spec["chapters"]} · runtime {spec["runtime"]}\n\n'
        f'---\n\n'
        f'{summary}\n\n'
        f'---\n\n'
        f'## How to use this notebook\n\n'
        f'1. Run the **Setup** cell — it defines the whole project but '
        f'runs nothing.\n'
        f'2. Run the project and read its output.\n'
        f'3. Work through the sections below, which take the pieces '
        f'apart.\n\n'
        f'Nothing is installed and nothing is downloaded: Colab already '
        f'has NumPy,\npandas, scikit-learn and PyTorch, and the data is '
        f'generated or bundled.\n\n'
        f'The gates are declared *before* the run. When one fails, the '
        f'useful\nquestion is which is wrong — the code, or the threshold '
        f'written down\nbefore the problem was understood.'
    )]

    cells.append(markdown_cell('## Setup\n\nDefines everything. Runs '
                               'nothing.'))
    cells.append(code_cell(
        '#@title Setup — run this cell first (collapsed: it is the whole '
        'project)\n' + prepare_sources(spec)))

    cells.append(markdown_cell(
        '## Run it\n\nThe complete project, start to finish, with its '
        'gates at the end.'))
    cells.append(code_cell(f'main({spec["run_args"]})'))

    for heading, blurb, code in spec['explore']:
        cells.append(markdown_cell(f'## {heading}\n\n{blurb}'))
        cells.append(code_cell(code))

    cells.append(markdown_cell(
        '---\n\n## Where this sits\n\nThe script version of this project '
        'is `ml-mastery/projects/' + spec['file'] + '`,\nand it writes an '
        'audit pack to `projects/runs/`. Part VI of the book walks\n'
        'through what to look for, including the gates that were wrong on '
        'the\nfirst run and why.'))

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
    """Generates every project notebook.

    Args:
        argv: Optional argument vector.

    Returns:
        0 on success.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=pathlib.Path, default=HERE)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for spec in SPECS:
        notebook = build_notebook(spec)
        path = args.out_dir / f'{spec["stem"]}.ipynb'
        path.write_text(json.dumps(notebook, indent=1), encoding='utf-8')
        size_kb = path.stat().st_size / 1024
        print(f'{path.name:<38} {len(notebook["cells"]):>2} cells, '
              f'{size_kb:>5.0f} KB')
    print(f'\n{len(SPECS)} project notebooks')
    return 0


if __name__ == '__main__':
    sys.exit(main())
