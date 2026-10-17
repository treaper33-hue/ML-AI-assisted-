# Colab Notebooks

The same 55 exercises as [`../exercises.py`](../exercises.py), in four
notebooks you can work in a browser. They are **generated from that
file**, so the gates are identical — there is one source of truth, and
the notebooks follow it.

| Notebook | Part | Exercises |
|---|---|---|
| [`part1_mathematical_engine.ipynb`](part1_mathematical_engine.ipynb) | I — The Mathematical Engine | 28 |
| [`part2_learning_from_data.ipynb`](part2_learning_from_data.ipynb) | II — Learning from Data | 12 |
| [`part3_neural_networks.ipynb`](part3_neural_networks.ipynb) | III — Neural Networks in PyTorch | 6 |
| [`part4_sequences_uncertainty.ipynb`](part4_sequences_uncertainty.ipynb) | IV — Sequences and Uncertainty | 9 |

## Opening them in Colab

They are **self-contained**: no clone, no data upload, no `pip install`.
Colab already ships NumPy and PyTorch, and nothing is read from disk.

1. Go to <https://colab.research.google.com>
2. **File → Upload notebook**, choose one of the four
3. Run the **Setup** cell once, then work down the page

If you push this repository to GitHub, you can skip the upload and open
them directly:

```
https://colab.research.google.com/github/<user>/<repo>/blob/main/notebooks/part1_mathematical_engine.ipynb
```

Substitute your own `<user>/<repo>`. There is deliberately no badge
hard-coded into the notebooks, because a link to a repository that does
not exist is worse than no link.

They run locally too — `jupyter notebook`, or VS Code — with the same
behaviour.

## How a notebook works

Each exercise is three cells:

1. **Markdown** — the task, and the pages to read first.
2. **A stub** — `raise NotImplementedYet(...)`, which you replace.
3. **`check('5.2')`** — runs that exercise's gate and prints `PASS`,
   `FAIL`, `TODO` or `SKIP`.

`check_all()` at the bottom summarises the notebook.

**A `TODO` is never a pass.** And when a gate fails, the threshold came
from the book, so the useful question is which of the two is wrong —
your code, or your understanding of the threshold. Both happen.

The Setup cell is collapsed because it contains the reference
solutions: two gates compare against them, so they have to be in the
runtime. Reading them before you have tried is a way of not learning the
material.

## Regenerating

After editing `../exercises.py`:

```bash
python make_notebooks.py        # rebuild all four
python validate_notebooks.py    # prove they still run
```

`validate_notebooks.py` executes every cell twice: once as shipped
(every gate must report `TODO`, never `FAIL` — a stub that fails is
broken rather than merely unfilled), and once with the reference
implementations bound in (every gate must then `PASS`, or it is a gate
nobody could ever satisfy). Current state:

```
4 notebooks execute cleanly, and every gate is satisfiable
```

---

# Project notebooks

The five worked projects of Part VI, as notebooks. Same principle: they
are generated from the scripts in [`../projects/`](../projects/), so
they cannot drift from them.

| Notebook | Project | Chapters |
|---|---|---|
| [`project1_tabular_churn.ipynb`](project1_tabular_churn.ipynb) | Tabular classification, end to end | 9–14 |
| [`project2_digits_cnn.ipynb`](project2_digits_cnn.ipynb) | A CNN trained by hand | 16–19 |
| [`project3_forecast_conformal.ipynb`](project3_forecast_conformal.ipynb) | Forecasting with real intervals | 22–25 |
| [`project4_vae_elbo.ipynb`](project4_vae_elbo.ipynb) | A VAE with the ELBO taken apart | 5, 8, 20 |
| [`project5_serving_monitoring.ipynb`](project5_serving_monitoring.ipynb) | Serving and monitoring | 28 |

Each has the same shape:

1. **Setup** — defines the entire project and *runs nothing*. A notebook
   that starts training the moment you open it is a bad notebook.
2. **Run it** — the project end to end, gates and all.
3. **Three exploration sections** — the pieces taken apart, with
   something to change and re-run in each.

Project 5 imports from Project 1 when run as a script; in the notebook
Project 1's source is embedded ahead of it, so it still needs no files.

## Regenerating and validating

```bash
python make_project_notebooks.py
python validate_project_notebooks.py            # fast: compile + setup + one run
python validate_project_notebooks.py --run-all  # every project, ~5 minutes
```

The validator checks what is specific to the notebook form, which is
where breakage actually lives: every cell compiles, the Setup cell
defines `main` without printing anything, and `__file__` is handled —
a notebook cell has none, and the projects use it to locate `runs/`.

Both bugs it caught were real. The first generated notebooks would have
greeted every reader with an `IndentationError`, and the projects would
have raised `NameError` on `__file__`. Neither is visible by reading.

Current state, with `--run-all`:

```
project1  main() returned 0 in 153.9s, VERIFIED
project2  main() returned 0 in  53.2s, VERIFIED
project3  main() returned 0 in   9.4s, VERIFIED
project4  main() returned 0 in  78.0s, VERIFIED
project5  main() returned 0 in  23.1s, VERIFIED
```

(The run-cell in each notebook uses the quick settings, so opening one
and running it is much faster than those figures.)
