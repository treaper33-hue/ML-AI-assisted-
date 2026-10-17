---

# Appendix A · What these sources do not cover

Stating the edge of your competence is part of having it. Four areas are not
covered by the sources behind this text, and nothing here should be read as
teaching them:

| Gap | What is missing | What would close it |
|---|---|---|
| **Reinforcement learning** | No RL theory in any of the four books. PPO and RLHF appear only as implementations in `papers/29_PPO` and `papers/30_RLHF`. | Sutton & Barto; the original PPO and RLHF papers. |
| **Distributed training** | No DDP, FSDP or ZeRO. The nearest material is PyTorch Lightning *[PTG p.351]* and GPipe in `papers/24_gpipe`. | A dedicated distributed-training source and a multi-GPU machine. |
| **Statistical learning theory** | No PAC learning, VC dimension, Rademacher complexity or concentration inequalities. *[MML]* stops at the law of large numbers *[p.595]* and information theory *[p.606]*. | Shalev-Shwartz & Ben-David, or Mohri. |
| **Production engineering** | *[PTG p.385–407]* covers local, Heroku and Azure. No containers, CI/CD, feature stores, or registries beyond MLflow. Project 5 covers serving, validation, latency, drift and rollback, but none of the infrastructure around them. | A dedicated MLOps source. |

Two further limitations of this text itself. The scikit-learn source is an
EPUB with no stable pagination, so its anchors are chapter and section
headings rather than pages. The *Time Series with PyTorch* anchors are PDF
page indices of the supplied file, not printed-book pages.

---

# Appendix B · The exercises

Every gate in this book is implemented in `exercises.py`, which sits beside
this PDF. It is one file, it runs on NumPy alone, and it reports each exercise
as `PASS`, `FAIL`, or `TODO` — the last meaning you have not implemented it
yet, which is never the same as passing.

```bash
python exercises.py                 # run everything
python exercises.py --part 1        # one part
python exercises.py --exercise 5.2  # one exercise
python exercises.py --list          # the index
python exercises.py --solutions     # run the reference implementations
```

Work in order. A skipped `TODO` is not a pass, and the summary at the end of
the run will tell you so in exactly those terms.

**55 exercises in total**, every one with a numeric gate taken from a chapter
of this text:

| Part | Exercises | Covers |
|---|---|---|
| 1 | 1.1 – 8.5 (27) | norms, Gram–Schmidt, LU, conditioning, eigenpairs, SVD, PCA, gradient descent and its stability limit, Jacobians, backprop, entropy, KL, MLE |
| 2 | 9.1 – 15.1 (12) | cost-based thresholds, the leakage demonstration, leak-safe scaling, ridge, the logistic gradient, soft-thresholding, Gini splits, precision/recall, bootstrap intervals, k-means |
| 3 | 17.1 – 21.1 (6) | stable softmax, the p − y gradient, convolution by im2col and its arithmetic, attention, a PyTorch training step |
| 4 | 22.1 – 25.2 (10) | autocorrelation, decomposition, rolling-origin folds, MASE and its invariance, APE at zeros, conformal coverage, the Winkler score |

Several gates reproduce a worked example from the text exactly — the Gini
gains of 0.18 and 0.056, the MASE of 0.625, the eigenvalues 5 and 2 — so when
one fails you can find the arithmetic on the page and compare line by line.

---

# Appendix C · The worked projects

Five complete systems in `ml-mastery/projects/`, described in Part VI.

```bash
cd ml-mastery/projects
python run_all.py              # all four, with a gate summary
python p3_forecast_conformal.py --no-shift    # the stationary control
```

| File | Chapters | Gates | Runtime |
|---|---|---|---|
| `p1_tabular_churn.py` | 9–14 | 6 | ~40 s |
| `p2_digits_cnn.py` | 16–19 | 8 | ~10 s |
| `p3_forecast_conformal.py` | 22–25 | 7 | ~2 s |
| `p4_vae_elbo.py` | 5, 8, 20 | 10 | ~20 s |
| `p5_serving_monitoring.py` | 28 | 8 | ~9 s |

Each writes an audit pack to `projects/runs/<name>/report.json`. None
downloads anything: data is generated or comes from scikit-learn's
bundled digits, and Project 5's server binds only to loopback.

---

# Appendix D · Working the exercises in Colab

The 55 exercises exist in two forms. `exercises.py` runs them all from a
terminal; four notebooks in `ml-mastery/notebooks/` run the same gates in
a browser. The notebooks are **generated from `exercises.py`**, so the
thresholds cannot drift apart — there is one source of truth and the
notebooks follow it.

| Notebook | Part | Exercises |
|---|---|---|
| `part1_mathematical_engine.ipynb` | I | 28 |
| `part2_learning_from_data.ipynb` | II | 12 |
| `part3_neural_networks.ipynb` | III | 6 |
| `part4_sequences_uncertainty.ipynb` | IV | 9 |

They are self-contained: no clone, no data upload, no `pip install`.
Colab ships NumPy and PyTorch, and nothing is read from disk.

```
1. https://colab.research.google.com
2. File -> Upload notebook, choose one of the four
3. Run the Setup cell once, then work down the page
```

`python exercises.py --colab` prints the same instructions, and
`--list` now names the notebook for each part.

If you push this repository to GitHub you can open them without the
upload step, at
`colab.research.google.com/github/<user>/<repo>/blob/main/ml-mastery/notebooks/<file>.ipynb`.
No badge is baked into the notebooks, because a link to a repository
that does not exist is worse than no link at all.

## The shape of a notebook

Each exercise is three cells: the task and its page anchors, a stub
raising `NotImplementedYet`, and a `check('5.2')` call that runs the
gate. `check_all()` summarises at the bottom. The Setup cell is
collapsed because it carries the reference solutions — two gates compare
against them, so they must be in the runtime.

## Why the notebooks are tested

A notebook that has never been executed is a liability, so
`validate_notebooks.py` runs every cell of every notebook twice:

- **as shipped** — every gate must report `TODO`, never `FAIL`. A stub
  that *fails* is broken rather than merely unfilled, and the two are
  not the same condition.
- **as completed** — the reference implementations are bound to the
  names the exercises expect, and every gate must then `PASS`. Without
  this pass a notebook could ship a gate nobody could ever satisfy.

Both passes are green for all four notebooks. Writing that validator
found a real defect, too: the first generated notebooks flattened each
stub's indentation onto its `def` line and would have greeted every
reader with an `IndentationError`. Nobody would have noticed by looking.

---


## The projects, too

The five projects of Part VI have notebooks as well, generated from the
scripts in `ml-mastery/projects/`:

| Notebook | Project | Chapters |
|---|---|---|
| `project1_tabular_churn.ipynb` | Tabular classification | 9–14 |
| `project2_digits_cnn.ipynb` | A CNN trained by hand | 16–19 |
| `project3_forecast_conformal.ipynb` | Forecasting with intervals | 22–25 |
| `project4_vae_elbo.ipynb` | A VAE, ELBO taken apart | 5, 8, 20 |
| `project5_serving_monitoring.ipynb` | Serving and monitoring | 28 |

Each has the same three-part shape: a **Setup** cell that defines the
whole project and *runs nothing*; a cell that runs it end to end with
its gates; and three exploration sections that take the pieces apart and
give you something to change. Project 5 embeds Project 1's source, since
it builds on it.

`validate_project_notebooks.py --run-all` executes every one of them.
All five reach `VERIFIED` inside a notebook namespace.

## Two bugs the validators caught

Both were invisible to reading, and both would have met every reader on
the first cell.

The first generated exercise notebooks used `inspect.cleandoc` on each
stub, which dedents the first line and the remainder by *their own*
margin — flattening every function body onto its `def` line. Every
notebook would have opened with an `IndentationError`.

The project notebooks then raised `NameError: __file__`. A notebook cell
has no `__file__`, and the projects use it to locate their `runs/`
directory. The generator now defines it against the working directory.

A third, subtler one is worth recording because it was *not* a bug in
the notebooks. Validating with a bare `dict` as the namespace made
`dataclasses` fail on a string annotation, because it resolves those
through `sys.modules[cls.__module__].__dict__` and a dict has no module.
Jupyter and Colab register a real module; the validator now does too.
**The harness was unfaithful, not the artifact** — the same distinction
as §30.4 and §31.3, one layer further out.

---

*End of text. The gates are the point; go and fail a few.*
