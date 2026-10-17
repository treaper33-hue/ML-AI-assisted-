# Machine Learning — Program to Full Professional Capability

**A verifiable, source-anchored curriculum for code, training, and execution.**

| | |
|---|---|
| **Built from** | *Mathematics of Machine Learning* · *Prompting Scikit-Learn* · *PyTorch: The Practical Guide* · *Time Series with PyTorch* · *Google Python Style Guide* · this repo's 30 paper implementations |
| **Governed by** | `Algorithm Prompt.txt` (the Prompt Guidelines) |
| **Evidence** | [SOURCE_LEDGER.md](SOURCE_LEDGER.md) — every claim page-anchored |
| **Review** | [VERIFICATION.md](VERIFICATION.md) — status board, bug log, review protocol |
| **Testbed** | [testbed/](testbed/) — one command boots everything |
| **Length** | 24 weeks at 10–12 h/week (compressed track: 14 weeks at 20 h/week) |

---

## How this document implements the Prompt Guidelines

| Guideline | Where it lives here |
|---|---|
| 1. Extract algorithms, models and test parameters **with page numbers**; highlight gaps | [SOURCE_LEDGER.md](SOURCE_LEDGER.md) §1–§5 (algorithms), §6 (13 open gaps) |
| 2. Assign methodologies to **isolated agents**, unified by a **shared input field** and **single measurement tools** | §2 Operating Model — four roles; `mlkit.data` is the only loader, `mlkit.metrics` the only scorer |
| 3. An **independent review stage**; reviewers record bugs; unverified results carry **warning tags** | [VERIFICATION.md](VERIFICATION.md) — the Auditor role, the bug log, the four status tags |
| 4. **Controls** to tweak parameters, compare methodologies, and read the source | §3 Controls — one YAML config per experiment, `--sweep`, `--compare`, `--show-source` |
| 5. Start with **one document and one baseline experiment**, scale gradually | This document, plus `testbed/experiments/exp01_baseline.py` — nothing else until EXP-01 is green |
| 6. **Pack dependencies and clean data** so an outsider boots the testbed from scratch with one button | `testbed/setup.ps1` / `setup.sh`, pinned `requirements.txt`, generated datasets with fixed seeds |

> **The discipline this enforces:** *"a pretty visualization can easily turn out to
> be mathematically wrong."* Nothing in this program counts as learned because you
> read it or because the plot looked right. It counts when a numeric gate passes
> and an auditor signs it off.

---

# Part 0 — The Outcome Contract

"Full professional capability" is not a feeling. It is this list, each line
provable by an artifact in `runs/`.

## 0.1 Competency matrix

| # | Competency | Proof artifact | Primary sources |
|---|---|---|---|
| C-01 | Derive and implement the linear algebra under ML from scratch | `mlkit/linalg.py` + passing tests for Gram–Schmidt, LU, SVD, QR | [MML p.67–p.230] |
| C-02 | Derive and implement gradient-based optimisation from scratch | `mlkit/optim.py`, convergence/divergence study at lr vs 2/L | [MML p.381, p.481] |
| C-03 | Reason probabilistically: distributions, entropy, KL, MLE | `mlkit/prob.py` + MLE recovery test | [MML p.534–p.619] |
| C-04 | Build a leak-proof classical ML pipeline and defend it | `experiments/exp02_tabular.py`, leakage test suite green | [SKL ch.3–ch.4], [TSP p.158–164] |
| C-05 | Evaluate honestly: right metric, right split, CI on every number | every `runs/*/report.json` carries CIs and the splitter used | [SKL ch.6], [TSP p.136–156] |
| C-06 | Train neural networks in PyTorch without framework magic | hand-written training loop matching autograd to 1e-6 | [PTG p.45–p.80] |
| C-07 | Implement a paper from the original text | one [30u30] day reproduced against the paper's own number | [30u30], [PTG p.320] |
| C-08 | Model sequences and forecast with correct temporal validation | `experiments/exp04_forecast.py`, rolling-origin folds asserted | [TSP p.126–p.156, ch.8–9] |
| C-09 | Quantify uncertainty with coverage guarantees | conformal interval with empirical coverage within 2% of target | [TSP p.309–p.346] |
| C-10 | Tune and scale training deliberately | LR-range test, batch sweep, Lightning port with identical metrics | [TSP ch.6], [PTG p.72, p.352] |
| C-11 | Track, reproduce and hand over experiments | audit pack a stranger can rerun and match | [SKL ch.14], [PTG p.363–p.377] |
| C-12 | Serve a model and measure it in production shape | local service with p50/p95 latency under load | [PTG p.385–p.407] |
| C-13 | Explain and govern a model | permutation importance + PDP/ICE + model card + incident plan | [SKL ch.11, ch.14] |
| C-14 | Write code a professional team would merge | `mlkit/` passes lint, types, and the 80-column contract | [GPSG p.1–p.55] |

**Not covered by this stack — do not self-certify** (see ledger §6):
reinforcement learning theory (GAP-04), distributed training (GAP-05),
statistical learning theory (GAP-06), container/CI-CD-grade production (GAP-07).
Those competencies stay tagged `UNVERIFIED` until an external source closes them.

## 0.2 Exit criteria

You are done when all of the following are true:

1. All 14 competencies have a `VERIFIED` row on the status board.
2. The capstone runs end to end from a clean clone with one command.
3. An independent reader — a person who has not seen your work — reruns the
   capstone and reproduces every headline number within its stated CI.
4. Every remaining gap is **named in writing**, not silently skipped.

---

# Part 1 — Before Anything Else

## 1.1 Boot the testbed (the one button)

```powershell
cd ml-mastery\testbed; .\setup.ps1
```

This creates `.venv`, installs pinned dependencies, generates the clean
datasets with fixed seeds, and runs the baseline experiment. It prints a status
line per arm. **Do not start Stage 1 until it prints `EXP-01 ... VERIFIED` for at
least the NumPy arm.**

To use an environment you already have — a conda env, say — pass it instead:

```powershell
cd ml-mastery\testbed; .\setup.ps1 -CondaEnv qwen35
```

That mode installs only what is absent and never changes a version you already
have, because downgrading a working environment to match our pins would break
more than it fixes. The trade-off is that pins are not enforced there, so its
numbers reproduce on that machine rather than from a clean clone (ledger
GAP-14). Both modes write the full environment into every run report, so the
distinction is never lost.

Why this comes first: Guideline 6 — an outside researcher must bring the whole
testbed up from scratch with a single button. If you cannot, nobody can check
your work, and unverifiable work is not professional work.

## 1.2 Read the baseline experiment before writing any code

`testbed/experiments/exp01_baseline.py` is deliberately small: fit the same
linear model three ways — closed-form normal equations, hand-written gradient
descent, and (when installed) scikit-learn and PyTorch — then score all arms with
**one** metrics module and compare them against the analytic optimum.

It is the template for every later experiment: same config object, same loader,
same scorer, same status tags, same report file.

---

# Part 2 — Operating Model

## 2.1 The four roles (Guideline 2)

You will play all four, but **never two at once**, and never two on the same
artifact in the same sitting. Separation is the point: the Coder's optimism is
exactly what the Auditor exists to cancel.

| Role | Mandate | Output | Forbidden |
|---|---|---|---|
| **Analyst** | Pull the mathematics, assumptions and test parameters out of the text. | A ledger row with a page anchor and an acceptance threshold. | Writing implementation code. |
| **Coder** | Turn the ledger row into working code under the style contract. | A module plus its unit tests, lint-clean and typed. | Choosing its own metric, or editing the acceptance threshold. |
| **Auditor** | Re-derive the result independently and try to break it. | A `VERIFIED` / `PENDING` / `UNVERIFIED` tag and bug-log entries. | Fixing the bug they found (hand it back). |
| **Interface** | Build the controls: config, sweeps, comparison tables, plots. | A reproducible report. | Changing numbers to make a plot look better. |

**Role-switch rule:** take a break, or sleep, between Coder and Auditor on the
same artifact. If you audit code you wrote ten minutes ago, you are rereading
your intentions rather than the code.

## 2.2 The shared input field

**`mlkit.data.load_dataset(name, seed)` is the only way data enters an
experiment.** Not `pd.read_csv` in a notebook, not a path hard-coded in a script.
One loader means every methodology is compared on byte-identical inputs — the
precondition for the comparison tables in §3.

Every dataset is either generated from a fixed seed or shipped with a checksum.
The loader returns data plus a `DatasetCard` recording name, shape, seed,
checksum, split policy and provenance. The card goes into the run report.

## 2.3 The single measurement tools

**`mlkit.metrics` is the only scorer.** Models never compute their own numbers.
This closes GAP-09 — three sources define overlapping metrics and they disagree
at the edges:

- APE is **undefined at zeros** [TSP p.144] — the implementation raises rather
  than silently returning `inf`.
- MAPE **changes under a unit change** (Celsius vs Kelvin, [TSP p.143]) — MASE
  does not; there is a unit-change test proving it.
- MASE/RMSSE denominators use the **naive forecast on train only** [TSP p.150] —
  computing them on test is a leak.
- Macro and micro F1 answer different questions [SKL ch.6] — both are reported,
  never one silently.

## 2.4 Status tags (Guideline 3)

Every result carries exactly one:

| Tag | Meaning |
|---|---|
| **VERIFIED** | Reproduced by the Auditor from a clean run, numeric gate passed. |
| **PENDING REVISION** | A gate failed or a bug is open. The number stays visible; it may not be cited. |
| **UNVERIFIED** | Ran, but has **no numeric gate** — a visual result, a missing dependency, a competency outside the source stack. Must be labelled wherever it appears. |
| **BUG** | A defect with a bug-log ID. Blocks the stage gate. |

An untagged number does not exist. If a plot appears in a report without the tag
of the number behind it, the Interface role has failed.

## 2.5 The loop

**Daily (about 2 h):**
1. *Analyst* 30 min — read the anchored pages, write the ledger row with its
   acceptance threshold **before** coding.
2. *Coder* 60 min — implement, test, lint.
3. *Auditor* 20 min — run the gate, tag, log bugs.
4. *Interface* 10 min — update the status board; commit the run.

**Weekly (about 1 h):** reread the week's bug log; close or escalate every open
item; reconcile the status board with `runs/`; write three sentences on what you
got wrong. The last part matters most — it is the only part that compounds.

---

# Part 3 — Controls (Guideline 4)

Every experiment is driven by a config, never by edited source. Configs are
TOML so the standard library's `tomllib` can read them — one fewer dependency
between an outside researcher and a working testbed.

```toml
# configs/exp01.toml
experiment = "exp01_baseline"
seed = 20260101

[dataset]
name = "synthetic_linear"
n_samples = 2000
n_features = 8
noise = 0.25

[split]
kind = "random"
test_size = 0.2

# Acceptance thresholds, declared BEFORE the code ran.
[gates]
coef_recovery_tol = 0.1
gd_agreement_tol = 1e-6

[arms.gradient_descent]
lr = 0.05
iters = 2000
momentum = 0.0
```

| Control | Command | What it gives you |
|---|---|---|
| Run one experiment | `python -m experiments.exp01_baseline --config configs/exp01.toml` | one tagged report in `runs/` |
| Tweak a parameter | `... --set arms.gradient_descent.lr=0.3` | no source edit, the override recorded in the report |
| Sweep | `... --sweep arms.gradient_descent.lr=0.01,0.2,0.5,0.9,1.1` | parameter-vs-metric curve, including the divergence point [MML p.381] |
| Compare methodologies | `... --compare` | all arms on identical data and one scorer, with CIs and deltas from the analytic optimum |
| Read the source | `... --show-source gradient_descent` | prints the arm's implementation and its ledger anchors |
| Check the environment | `python -m mlkit.verify --env` | what this machine can and cannot verify, with broken installs called out separately from absent ones |
| List what is absent | `python -m mlkit.verify --missing` | pip names only, for installing into an existing environment |
| Re-check the gates | `python -m mlkit.verify runs/<id>` | re-reads a finished run and prints its tag |
| See every run at once | `python -m panel.serve` (from `testbed/`) | read-only dashboard over `runs/`: status board, per-run arms, gates with their anchors, splits and gaps, config-diff-beside-metric comparison, one-key sweeps with divergence kept off the axis, and a standing banner for every run that is not `VERIFIED`. No metric is drawn without the status tag of the arm that produced it, and nothing in it can edit a run, a threshold or a status |

Every run writes `runs/<timestamp>-<experiment>/` containing `config.json` (the
resolved config, overrides included), `report.json` (which carries `created_at`,
so the pack states its own run time rather than leaving it in the directory
name), `report.md`, `env.txt`
(interpreter, package versions, platform), `dataset_card.json`, and the plots. That directory **is** the audit pack
[SKL ch.14 "What to Include in an Audit Pack"]. `python -m panel.serve` reads
those directories and nothing else; the panel is a viewer over the packs, never
a second place a result can live.

---

# Part 4 — The Curriculum

Each module states: **Objective · Anchors · Build · Train · Execute · Gate ·
Controls · Blind spots.** The Gate is the only thing that advances you.

---

## Stage 0 — Testbed and Craft
*Week 0 · about 10 h · Competencies C-14, C-05 (foundation)*

### M0.1 — Boot and verify the environment
**Objective.** A stranger can reproduce you.
**Build.** Run `setup.ps1`; read what it does line by line.
**Gate.** `python -m mlkit.verify --env` prints interpreter, versions and
platform, and `EXP-01` reports `VERIFIED` on the NumPy arm.
**Blind spots.** GAP-13 — the system interpreter has only NumPy and matplotlib;
every arm needing scikit-learn or PyTorch is `UNVERIFIED` until the venv exists.
GAP-12 — no runtime baselines exist in any source, so record your own.

### M0.2 — The code contract
**Objective.** Write code a professional team merges without comment.
**Anchors.** [GPSG p.1 §2.1] lint · [p.19 §2.21] and [p.46 §3.19] type
annotations · [p.20 §3.2] 80 columns · [p.27 §3.8] docstrings · [p.43 §3.16]
naming · [p.45 §3.17] `main()` guard · [p.46 §3.18] function length.
**Build.** Read `mlkit/metrics.py` against that list and find the deliberate
style breaks — there are none; prove it with the linter.
**Gate.** `ruff check mlkit && ruff format --check mlkit` clean at line-length 80.
**Blind spots.** GAP-10 — [GPSG] covers style, not ML hygiene; seeds, configs and
data versioning come from Appendix B of this document and have no external
authority.

### M0.3 — The measurement contract
**Objective.** One definition per metric, forever.
**Anchors.** [TSP p.138] AE/MAE · [p.139] APE · [p.143] the unit-change table ·
[p.144] APE at zeros · [p.150] MASE/RMSSE · [SKL ch.6 "Task-Appropriate Metrics"].
**Build.** Read `mlkit/metrics.py`; add one metric yourself (sMAPE) with its
edge-case test.
**Gate.** `pytest tests/test_metrics.py` green from the testbed directory,
including the zero-division and unit-change tests.
**Blind spots.** GAP-09 — if you ever compute a metric outside this module, every
cross-methodology comparison in the program becomes meaningless.

### M0.4 — The leakage contract
**Objective.** Make leaks impossible rather than unlikely.
**Anchors.** [TSP p.158] leakage · [p.159] transformation-based · [p.161]
lookforward bias · [p.163] panel leakage · [p.164] the prevention checklist ·
[SKL ch.3 "Leakage-Safe Evaluation"].
**Build.** `mlkit/splits.py` ships random, fixed-origin, expanding and rolling
splitters, each returning indices plus a human-readable boundary description.
Add stratified and group splitters yourself.
**Train.** Run `pytest tests/test_leakage.py -s`. It selects features on pure
noise using all the data and then using the training rows only, and prints both
scores. On the shipped seed the leak manufactures R² = 0.35 out of data with no
signal in it, while the honest fit correctly lands below zero. Record the number
— it is the most useful number in Stage 0.
**Gate.** The leak suite passes, and you can explain each of its three failure
modes in one sentence without looking.
**Blind spots.** GAP-08 — three sources disagree; [TSP p.158–164] is canonical
here, including for non-temporal work.

### M0.5 — Run discipline
**Objective.** Every result is recoverable from its directory alone.
**Anchors.** [SKL ch.14 "What Makes Runs Reproducible", "What to Include in an
Audit Pack"] · [PTG p.363 TensorBoard, p.372 MLflow].
**Gate.** Delete your working directory's `__pycache__`, rerun a `runs/` entry
from its own `config.json`, and reproduce the metrics bit for bit at a fixed seed.

**STAGE 0 GATE:** EXP-01 `VERIFIED` · lint and types clean · metric and leak tests
green · one reproduced run.

---

## Stage 1 — The Mathematical Engine
*Weeks 1–4 · about 45 h · Competencies C-01, C-02, C-03*

Rule for this stage: **implement before you use**. Every routine here exists in
NumPy already; writing it yourself is the point, and the NumPy version is your
test oracle.

### M1.1 — Vectors, norms, inner products
**Anchors.** [MML p.9] vector spaces · [p.14] basis · [p.25–p.37] vectors in
practice, including "Is NumPy really faster than Python?" [p.37] · [p.49] norms ·
[p.56] inner products · [p.67] Gram–Schmidt theory · [p.89] Gram–Schmidt in code ·
[p.71] orthogonal complement.
**Build.** `lp_norm`, `cosine_similarity`, `gram_schmidt`, `project_onto`.
**Train.** The vectorisation benchmark [p.37]: pure Python vs NumPy at
n ∈ {1e3, 1e4, 1e5, 1e6}.
**Gate.** QᵀQ = I to 1e-10; norms match `np.linalg.norm` to 1e-12; the benchmark
table is in your run report (no target constant — GAP-12).
**Blind spots.** Classical Gram–Schmidt loses orthogonality on ill-conditioned
input; check κ(A) and switch to modified Gram–Schmidt. [MML] presents the
classical form — this caveat is **locally authored**, so tag it `UNVERIFIED`
against the source until you confirm it numerically yourself (you will: build a
Hilbert matrix and watch it fail).

### M1.2 — Linear maps, determinants, solving systems
**Anchors.** [MML p.117] linear transformations · [p.128] change of basis ·
[p.133–p.138] the plane-transform gallery · [p.139] determinants and volume ·
[p.160] Gaussian elimination · [p.165] its complexity · [p.168] LU · [p.171] LU in
code · [p.177] "How to actually invert matrices" · [p.178] determinants in practice.
**Build.** `gauss_solve`, `lu`, `det_via_lu`, plus the 2×2 transform gallery.
**Train.** Time Gaussian elimination at n ∈ {64, 128, 256} and fit the exponent;
time recursive vs LU determinants until recursion becomes unusable.
**Gate.** ‖PA − LU‖ < 1e-10; the fitted exponent is near 3 [p.165]; your solver
and `np.linalg.solve` agree to 1e-10.
**Blind spots.** Explicit inversion is numerically worse than solving [p.177] —
demonstrate it on a near-singular system rather than taking it on faith.

### M1.3 — Eigenstructure and factorisations
**Anchors.** [MML p.189] eigenvalues · [p.192] characteristic polynomial ·
[p.203] spectral decomposition · [p.207] **SVD** · [p.210] orthogonal projections ·
[p.217] computing eigenvalues · [p.230] **QR algorithm** · [p.243–p.251] matrices
and graphs.
**Build.** `power_iteration`, `qr_algorithm`, `svd_lowrank`.
**Train.** Low-rank image or matrix approximation across k; plot error vs k
against the Eckart–Young prediction.
**Gate.** Rank-k reconstruction error equals the tail singular-value norm to
1e-10; QR converges on a symmetric matrix with the iteration count logged.
**Blind spots.** The characteristic polynomial is a teaching device, not a
numerical method past tiny n — [p.217] and [p.230] exist precisely because of
that. Say so in your notes.

### M1.4 — Calculus and one-dimensional optimisation
**Anchors.** [MML p.287] sequences and series · [p.321] limits and continuity ·
[p.343] differentiation · [p.372] minima and maxima · [p.381] **the basics of
gradient descent** · [p.389] why gradient descent works · [p.405–p.418]
integration.
**Build.** `central_difference`, `gd`, `trapezoid`, `simpson`.
**Train.** The learning-rate study: lr ∈ {0.001 … 2.5} on a quadratic with known
Lipschitz constant L.
**Gate.** Convergence for lr < 2/L and **documented divergence above it** — the
divergence is a required result, not a failure.
**Blind spots.** Numerical differentiation trades truncation against round-off;
find the h that minimises total error empirically instead of using a rule of thumb.

### M1.5 — Gradients in many variables
**Anchors.** [MML p.435] multivariable functions · [p.443] **the curse of
dimensionality** · [p.448] partial and total derivatives · [p.460] vector-valued
derivatives · [p.473] multivariable functions in code · [p.481] **gradient descent
in its full form**.
**Build.** `jacobian`, `gd_nd` with a momentum toggle.
**Train.** Rosenbrock with and without momentum; the dimensionality experiment
from [p.443] at d ∈ {2 … 100}.
**Gate.** Numeric and analytic Jacobians agree to 1e-6; Rosenbrock reaches
‖∇f‖ < 1e-4 with the iteration counts for both settings reported.
**Blind spots.** This is backpropagation in miniature. If the chain rule here is
hazy, [PTG p.45] will stay magic — do not advance on a hazy Jacobian.

### M1.6 — Probability, information, likelihood
**Anchors.** [MML p.498] axioms · [p.515] conditional probability · [p.534]
random variables · [p.542] discrete distributions · [p.556] real-valued
distributions · [p.567] densities · [p.580–p.589] expected value and variance ·
[p.595] **law of large numbers** · [p.606] **information theory** · [p.619] **MLE**.
**Build.** `bayes_update`, samplers, `welford_variance`, `entropy`,
`kl_divergence`, `mle_gaussian`, `mle_bernoulli`.
**Train.** LLN convergence plot; MLE recovery at n ∈ {1e2 … 1e5}.
**Gate.** KL(p‖p) = 0 and KL ≥ 0 across 1000 random pairs; MLE recovers μ and σ²
within 3 standard errors; sample-mean error shrinks as O(n^-1/2).
**Blind spots.** Cross-entropy loss in [PTG p.38] **is** the KL machinery from
[p.606] — write the derivation connecting them in your notes. GAP-06: no
generalisation bounds exist in this stack; you can compute, not bound.

**STAGE 1 GATE (C-01, C-02, C-03):** all `mlkit` math tests green; the
learning-rate divergence study is in `runs/`; EXP-01 extended so that the
hand-written gradient-descent arm matches the normal-equation solution to within
1e-6 — proving your calculus and your linear algebra agree.

---

## Stage 2 — Classical ML with Honest Evaluation
*Weeks 5–8 · about 45 h · Competencies C-04, C-05*

### M2.1 — Framing and schema
**Anchors.** [SKL ch.1 "The Main Types of ML Problems", "Choosing Metrics and
Thresholds That Match with the Goal", "Error Analysis and Slice Checks"] ·
[ch.3 "Foundations: ... Read Data as a Schema", "Data Types on Disk at Load Time"].
**Build.** A framing card and a `schema.yaml` per dataset; a Parquet round-trip.
**Gate.** The metric is chosen and written down **before the first fit**, with the
cost of being wrong stated in one sentence.
**Blind spots.** The most expensive ML mistakes are framing mistakes, and no gate
catches them. This is where a human reviewer is irreplaceable.

### M2.2 — Pipelines that cannot leak
**Anchors.** [SKL ch.3 "The Preprocessing Toolkit", "Putting It Together: One
Leak-Proof ColumnTransformer", "End-to-End Pipelines"] · [TSP p.158–p.164].
**Build.** One `ColumnTransformer` + `Pipeline` covering imputation, encoding,
scaling, dates and text.
**Train.** Score the pipeline under CV; score the bare model with preprocessing
done up front; report the gap.
**Gate.** The gap is positive and explained in writing.
**Blind spots.** Group and time leakage survive a correct `ColumnTransformer`
[TSP p.161, p.163] — the splitter is a separate defence.

### M2.3 — Supervised models
**Anchors.** [SKL ch.4 "Linear and Generalized Linear Models", "Trees and
Ensembles", "Ideal Use-Cases for Each Model"] · ridge ties back to [MML p.210].
**Build.** Ridge from scratch (normal equations with a penalty), then logistic
regression, trees, random forest, gradient boosting through sklearn.
**Gate.** Your ridge matches sklearn's coefficients to 1e-6; every model reports a
learning curve, not a single score.
**Blind spots.** A random forest's defaults beat a badly tuned boosted model, and
that comparison teaches nothing — compare at matched budgets.

### M2.4 — Evaluation, calibration, intervals
**Anchors.** [SKL ch.6] all of: "Task-Appropriate Metrics", "Cross-Validation
Patterns", "Nested CV", "Learning Curves and Validation Curves", "Uncertainty and
Robustness", "Probability Calibration" · [TSP p.136–p.156] for the metric algebra.
**Build.** `nested_cv`, `bootstrap_ci`, reliability diagrams.
**Train.** Flat CV vs nested CV on the same search space.
**Gate.** Nested ≤ flat, with the optimism gap reported; every headline number
carries a 95% CI; Brier score improves after calibration or you explain why not.
**Blind spots.** PR-AUC and ROC-AUC rank models differently under class imbalance
[SKL ch.6] — report both and say which decides.

### M2.5 — Feature engineering and selection
**Anchors.** [SKL ch.9 "Constructing Features That Generalize", "Datetime:
Snapshot Rules, Lags, and Rollups (without leakage)", "Selecting Features", the
two case studies] · lag mechanics cross-check [TSP p.161].
**Gate.** Selection happens **inside** the CV loop; prove the inflation when it
does not by running it both ways.
**Blind spots.** Feature selection outside the fold is the single most common
silent leak in tabular ML.

### M2.6 — Tuning
**Anchors.** [SKL ch.10 "Leak-Safe Search over the Whole Pipeline", "Search
Strategies and Space Design", "Hyperparameter Glossary", "Time-Aware Search"].
**Build.** `search_space.yaml`; random search, then a coarse-to-fine refinement.
**Gate.** The search runs over the pipeline; the seed is fixed; the test set is
untouched until the end.
**Blind spots.** More search on a small validation set buys overfitting, not
performance — watch the gap between selection score and test score.

### M2.7 — Unsupervised structure
**Anchors.** [SKL ch.5 "Clustering Algorithms", "Dimensionality Reduction",
"Evaluation and Diagnostics"] · PCA derivation from [MML p.207].
**Build.** PCA from your own SVD, then KMeans/DBSCAN/Agglomerative sweeps.
**Gate.** Your PCA matches sklearn's up to sign; cluster reports include stability
across ≥ 5 seeds.
**Blind spots.** t-SNE and UMAP are **visualisation, not features** [SKL ch.5];
using their output as model input is a documented error. Silhouette rewards
spherical clusters, so it will mislead you on DBSCAN.

**STAGE 2 GATE (C-04, C-05):** `experiments/exp02_tabular.py` runs a full
leak-proof benchmark — at least four model families, nested CV, calibrated
probabilities, bootstrap CIs, slice analysis — and an Auditor reproduces it from
the run directory alone.

---

## Stage 3 — Deep Learning in PyTorch
*Weeks 9–13 · about 55 h · Competencies C-06, C-07*

### M3.1 — Tensors, autograd, the computation graph
**Anchors.** [PTG p.42] tensors · [p.43] creation and attributes · [p.45] **the
calculation graph and training** · [p.35–p.37] activations · [p.38–p.40] losses ·
[p.40] optimizers · cross-check [TSP ch.4 p.168].
**Build.** ReLU/tanh/sigmoid/softmax and MSE/BCE/CE by hand with their gradients.
**Gate.** Hand gradients match `torch.autograd` to 1e-6 on random inputs; your SGD
step matches `torch.optim.SGD` exactly.
**Blind spots.** Softmax overflows without the max-subtraction trick; [PTG p.37]
does not dwell on it — add the stability test yourself and tag it as locally
authored.

### M3.2 — The training loop, by hand
**Anchors.** [PTG p.52–p.59] data prep · [p.61–p.65] import/train/evaluate/infer ·
[p.68] the model class and optimizer · [p.72] **batches** · [p.76] **Dataset and
DataLoader** · [p.80] save and load · [p.83] sampling.
**Build.** A `Dataset`, a `DataLoader`, an `nn.Module`, and the loop with no
framework helpers.
**Train.** Batch-size sweep over {1, 8, 32, 256}: wall clock and final loss.
**Gate.** Deterministic order at a fixed seed; a reloaded checkpoint reproduces
logits bit for bit [p.80]; your parameter count matches your hand count.
**Blind spots.** Determinism needs the seed **and** worker settings **and**
sometimes cuDNN flags; "it's random" is not an acceptable explanation for a run
you cannot reproduce.

### M3.3 — Classification and diagnostics
**Anchors.** [PTG p.93] types · [p.95] confusion matrix · [p.97] ROC · [p.99]
binary · [p.112] multiclass · scoring via `mlkit.metrics` only.
**Gate.** Your by-hand confusion matrix matches `mlkit.metrics`; PR-AUC and
ROC-AUC both reported with bootstrap CIs; macro and micro F1 separated.
**Blind spots.** Accuracy on imbalanced data is a trap already set in Stage 2 —
the deep model does not change the statistics.

### M3.4 — Convolutional vision
**Anchors.** [PTG p.128] how models handle images · [p.129] architecture ·
[p.134] image classification · [p.163] object detection · [p.178] segmentation ·
[p.188] style transfer.
**Build.** Convolution by hand (im2col), then `F.conv2d`; IoU by hand.
**Gate.** Your convolution matches `F.conv2d` to 1e-5; IoU correct on known boxes;
a top-1 score with a CI.
**Blind spots.** Style transfer [p.188] has **no numeric gate** — it is permanently
`UNVERIFIED` in this program (ledger P-20). Enjoy it; do not cite it.

### M3.5 — Representation models
**Anchors.** [PTG p.202] recommenders · [p.220] autoencoders · [p.230, p.231]
**VAE** · [p.241–p.250] graph neural networks.
**Gate.** AE reconstruction beats a PCA baseline at equal latent dimension, or you
explain why it does not; the VAE's KL term matches the definition at [MML p.606].
**Blind spots.** A VAE whose KL collapses to zero has become an autoencoder —
monitor both ELBO terms separately, never just their sum.

### M3.6 — Implement papers from the original text
**Anchors.** [30u30 days 1–16] · [PTG p.320] how transformers work ·
[30u30 day 13] attention · [day 14] the annotated transformer.
**Build.** Work the repo's own loop: `README.md` → `paper_notes.md` →
`notebook.ipynb` → `implementation.py` → `exercises/`.
**Gate.** For at least one day, reproduce a number the **paper itself** reports —
not a number the repo reports.
**Blind spots.** GAP-11 — these implementations are educational and un-audited.
Every output stays `PENDING REVISION` until you reproduce it against the paper.
That is not pedantry: it is the one habit that separates reading papers from
knowing them.

**STAGE 3 GATE (C-06, C-07):** a deep model trained from a hand-written loop with
a reproducibility proof, plus one paper reproduced against its own reported number.

---

## Stage 4 — Sequences, Uncertainty, and Scale
*Weeks 14–18 · about 55 h · Competencies C-08, C-09, C-10*

This is the hardest stage and the one that most distinguishes a professional:
temporal data punishes every evaluation sin instantly.

### M4.1 — Temporal diagnostics
**Anchors.** [TSP p.52] missing data · [p.53] its types · [p.57] simple
imputation · [p.60] **MICE** · [p.65] decomposition · [p.66] additive · [p.70]
multiplicative · [p.73] STL · [p.76] seasonality plots · [p.84] **ACF** · [p.86]
**PACF** · [p.92] stationarity · [p.97] **ADF, KPSS, PP** · [p.103–p.106]
univariate / multivariate / multivariable / panel structures.
**Build.** ACF and PACF by hand; additive and multiplicative decomposition.
**Gate.** ACF/PACF match `statsmodels` to 1e-8; all three stationarity tests run
and **both outcomes are recorded when ADF and KPSS disagree** [p.97].
**Blind spots.** Imputation quality is not the objective — downstream metric is
[p.57–p.62]. Misclassifying the structure [p.103–p.106] guarantees the wrong
splitter downstream.

### M4.2 — Temporal evaluation
**Anchors.** [TSP p.113] why we evaluate · [p.119] partitioning · [p.126]
**fixed-origin** · [p.129] train/val/test · [p.132] **expanding windows** ·
[p.134] **rolling windows** · [p.136] NN validation shortcuts · [p.137–p.144]
intrinsic metrics · [p.143] the unit-change table · [p.149–p.150] **extrinsic
metrics, MASE, RMSSE** · [p.154] grouped · [p.155] multivariate and "which to use
and when" · [p.156] length of evaluation · [p.158–p.164] **leakage**.
**Build.** `fixed_origin_split`, `expanding_window_cv`, `rolling_window_cv`.
**Train.** Horizon sweep: metric vs h.
**Gate.** Fold boundaries printed and asserted; no training index exceeds any test
index; MASE's denominator provably uses train only; the metric-vs-horizon curve is
in the report.
**Blind spots.** [p.136] notes practitioners often use a single simple split for
neural networks — that is a **budget decision, not a correctness one**. If you take
it, write down that you took it.

### M4.3 — Neural forecasting baselines
**Anchors.** [TSP ch.5 p.202] simple architectures · [PTG p.261] modelling
approaches · [p.266] a custom model · [p.280] PyTorch Forecasting.
**Gate.** Your MLP forecaster beats seasonal-naive on MASE — **or the failure is
reported as the result**. A deep model that loses to naive is a finding, not an
embarrassment; hiding it is the only failure available here.
**Blind spots.** On short series, naive and ETS frequently win. This is the most
commonly suppressed result in applied forecasting.

### M4.4 — Optimisation craft
**Anchors.** [TSP ch.6 p.245] LR schedules, regularisation, activations, HPO ·
[PTG p.72] batches · [MML p.481] the underlying method.
**Train.** LR-range test; schedule comparison at a fixed budget.
**Gate.** A loss-vs-LR curve exists and your chosen LR is justified by it.
**Blind spots.** Tuning on the test set through repeated peeking is invisible and
fatal; the schedule choice belongs to validation only.

### M4.5 — Conformal prediction
**Anchors.** [TSP p.311] uncertainty quantification · [p.316] understanding
conformal prediction · [p.317] the intuition · [p.323] improving intervals ·
[p.325] quantiles · [p.331] arbitrary confidence levels · [p.332] **the
probability-bound caveat** · [p.333] bootstrap / cross-conformal / jackknife ·
[p.334] conformalised regression · [p.335] conformalised forecasting · [p.337]
**EnbPI** · [p.340] EnbPI on a PyTorch model · [p.346] interval quality.
**Build.** Split conformal, then EnbPI.
**Train.** Alpha sweep over {0.2, 0.1, 0.05, 0.01}.
**Gate.** Empirical coverage within 2% of 1 − α on held-out data; interval width
reported alongside — coverage without width is meaningless.
**Blind spots.** [p.332] warns the upper probability bound found in the
literature may not hold. Carry that warning into your report. Exchangeability is
violated under distribution shift, which is the normal condition of forecasting:
measure coverage **under shift**, not only in-sample.

### M4.6 — Recurrent and attention architectures
**Anchors.** [TSP ch.8 p.351] RNN/LSTM/GRU, dynamical-systems view · [ch.9 p.388]
transformers · [PTG p.320] the deep dive · [30u30 days 1–3] RNN, LSTM,
regularisation · [days 13–16] attention and transformers.
**Gate.** Your from-scratch LSTM's gradients match `nn.LSTM` to 1e-5; RNN and
transformer forecasters are benchmarked on **identical splits with identical
metrics**.
**Blind spots.** Transformers do not automatically beat an LSTM on small series;
the comparison is only meaningful at matched parameter and compute budgets.

### M4.7 — Global models and generative sequences
**Anchors.** [TSP ch.10 p.423] other structures (GNN, KAN, foundation models) ·
[ch.11 p.461] transfer learning and global models · [ch.12 p.514] synthetic data
and TimeVAE · [ch.13 p.555] diffusion.
**Gate.** Global model vs per-series models on pooled MASE, same folds.
**Blind spots.** Global models help on many short series and hurt on few long
heterogeneous ones — state which regime you are in before you conclude anything.

### M4.8 — Classification, clustering, embeddings, anomalies, self-supervision
**Anchors.** [TSP ch.14 p.586] classification via `aeon` · [ch.15 p.638]
clustering · [ch.16 p.676] embeddings · [ch.17 p.690] anomaly detection ·
[ch.18 p.763] self-supervised learning and CPC.
**Gate.** Anomaly detection reports precision@k against labelled anomalies, not
"the plot looks right"; the SSL module reports a label-efficiency curve.
**Blind spots.** Anomaly labels are usually scarce and subjective — say how yours
were produced, or your precision@k means nothing.

**STAGE 4 GATE (C-08, C-09, C-10):** `experiments/exp04_forecast.py` benchmarks
naive, classical, MLP, RNN and transformer arms on rolling-origin folds with MASE
and RMSSE, conformal intervals with verified coverage, and an LR study justifying
the optimiser settings.

---

## Stage 5 — Production and Professional Practice
*Weeks 19–22 · about 40 h · Competencies C-11, C-12, C-13*

### M5.1 — Transfer learning and fine-tuning
**Anchors.** [PTG p.329] Hugging Face · [p.332] transfer learning · [p.335]
fine-tuning a vision model · [p.343] fine-tuning a language model.
**Gate.** Frozen vs fine-tuned at an equal compute budget, with the delta and the
compute cost both reported.

### M5.2 — Training infrastructure
**Anchors.** [PTG p.351] PyTorch vs Lightning · [p.352] model training · [p.359]
callbacks.
**Gate.** A ported model produces **identical metrics** to your raw loop at the
same seed. If it does not, you have found a real bug — log it.
**Blind spots.** GAP-05 — nothing here covers DDP/FSDP; multi-GPU capability stays
`UNVERIFIED`.

### M5.3 — Tracking and monitoring
**Anchors.** [PTG p.363] TensorBoard · [p.372] MLflow · [p.377] Weights & Biases ·
[SKL ch.14 "What Makes Runs Reproducible"].
**Gate.** Delete a run's local artifacts and rebuild the result from the tracking
record alone.

### M5.4 — Deployment
**Anchors.** [PTG p.385] strategies · [p.387] local deployment · [p.393] Heroku ·
[p.399] Azure.
**Build.** A local inference service with input validation and a health check.
**Gate.** p50 and p95 latency under a fixed load, plus a documented rollback path.
**Blind spots.** GAP-07 — no containers, CI/CD, feature stores or registries in
these sources. This module is an **introduction**; the competency stays
`UNVERIFIED` until closed externally.

### M5.5 — Explainability
**Anchors.** [SKL ch.11 "Global Importance That Holds up: Permutation Importance",
"Directionality and Effect Size: Partial Dependence and ICE", "Interpretable
Baselines", "Slice Analysis", "Local 'Why This Prediction?'"].
**Gate.** Permutation importance computed on held-out data; PDP and ICE plotted
together (PDP alone hides heterogeneity); one slice where the model fails, named.

### M5.6 — Responsible AI and governance
**Anchors.** [SKL ch.14] scope and stakeholders, data governance, fairness and
harm, transparency, integrity and robustness, human-in-the-loop, monitoring and
incidents, reproducibility and the audit pack · [ch.15], [ch.16].
**Gate.** A model card, a fairness/harm assessment with a named protected
attribute, an incident plan with a rollback trigger, and an audit pack a stranger
can rerun.

### M5.7 — Frontier reading
**Anchors.** [30u30 day 24] GPipe · [day 25] scaling laws · [day 27] machine
super-intelligence · [day 29] PPO · [day 30] RLHF.
**Gate.** A one-page written argument on what scaling laws imply for a project you
would actually run.
**Blind spots.** GAP-04 — RL has no textbook support in this stack. Days 29–30 are
`UNVERIFIED` reading, not RL competence. Say so out loud; the ability to say it is
itself a professional signal.

**STAGE 5 GATE (C-11, C-12, C-13):** a tracked, served, explained and governed
model, with an audit pack reproduced by someone else.

---

# Part 5 — Capstone
*Weeks 23–24 · about 25 h · proves C-01 … C-14 together*

**Build one system that a team could inherit.** Recommended: a demand- or
load-forecasting service, because it exercises every stage.

Required components:

1. **Data** through `mlkit.data` with a dataset card, schema and checksum.
2. **Diagnostics** — decomposition, ACF/PACF, all three stationarity tests, with
   disagreements recorded [TSP p.97].
3. **Four arms minimum** — naive, classical, gradient-boosted tabular with lags,
   neural (RNN or transformer) — on identical rolling-origin folds [TSP p.132–134].
4. **One scorer** — MASE and RMSSE with bootstrap CIs [TSP p.150], everything from
   `mlkit.metrics`.
5. **Conformal intervals** with measured coverage and width, including under a
   deliberate distribution shift [TSP p.337–p.346].
6. **Leakage test suite** green [TSP p.158–p.164].
7. **Explainability** — permutation importance and PDP/ICE [SKL ch.11].
8. **A served endpoint** with latency numbers and a rollback path [PTG p.387].
9. **An audit pack** [SKL ch.14] plus a model card.
10. **A gap statement** — every `UNVERIFIED` tag in the system, listed, with what
    would close it.

**Capstone gate.** A person who has not seen your work clones the repository, runs
one command, and reproduces every headline number within its stated CI. If they
cannot, the capstone is `PENDING REVISION` — regardless of how good the model is.

---

# Part 6 — Schedules

## 6.1 Standard track — 24 weeks at 10–12 h/week

| Weeks | Stage | Headline deliverable |
|---|---|---|
| 0 | Testbed and Craft | EXP-01 verified, contracts in place |
| 1–4 | Mathematical Engine | `mlkit` math modules, LR divergence study |
| 5–8 | Classical ML | EXP-02 leak-proof tabular benchmark |
| 9–13 | Deep Learning | hand-written loop + one paper reproduced |
| 14–18 | Sequences and Uncertainty | EXP-04 forecasting with conformal coverage |
| 19–22 | Production | tracked, served, explained, governed model |
| 23–24 | Capstone | reproduced by an independent reader |

## 6.2 Compressed track — 14 weeks at about 20 h/week

Weeks 1–2 Stage 0 + Stage 1 (M1.1–M1.3, M1.5, M1.6; skim M1.4's theory but keep
the LR study) · 3–5 Stage 2 (keep M2.2, M2.4; compress M2.7) · 6–8 Stage 3 (keep
M3.1, M3.2, M3.6; M3.4 light) · 9–12 Stage 4 (no cuts — this stage is where the
professional distinction lives) · 13 Stage 5 (M5.3, M5.4, M5.6) · 14 Capstone.

**Never compressible:** the metric contract (M0.3), the leakage contract (M0.4),
temporal evaluation (M4.2), and conformal coverage (M4.5). Everything else can be
re-learned later; these four are the ones that make every other number true or
false.

## 6.3 The daily shape

| Minutes | Role | Activity |
|---|---|---|
| 30 | Analyst | read the anchored pages, write the ledger row and the threshold |
| 60 | Coder | implement, test, lint |
| 20 | Auditor | run the gate, tag the result, log bugs |
| 10 | Interface | update the board, commit the run |

---

# Appendix A — The style contract (enforced)

From [GPSG], applied to every file in `mlkit/` and `experiments/`:

```python
"""One-line summary of the module.

Longer description if the module needs one.
"""                                         # [GPSG p.27 §3.8.2]

import dataclasses                          # stdlib first  [p.40 §3.13]

import numpy as np                          # third party

from mlkit import metrics                   # local


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes the root mean squared error.

    Args:                                   # [p.29 §3.8.3]
        y_true: Ground-truth values, shape (n,).
        y_pred: Predicted values, shape (n,).

    Returns:
        The RMSE as a float.

    Raises:
        ValueError: If the shapes do not match.
    """
```

- 80 columns [p.20 §3.2] · 4 spaces [p.23 §3.4] · `module_name`, `ClassName`,
  `function_name`, `GLOBAL_CONSTANT`, `_internal` [p.43 §3.16.2].
- Type annotations on every public function [p.19 §2.21, p.46 §3.19].
- `main()` plus the `if __name__ == '__main__':` guard in every executable
  [p.45 §3.17].
- Functions past ~40 lines get rethought [p.46 §3.18].
- `TODO(username): ...` [p.39 §3.12].
- **Documented exemption:** mathematical notation may use short local names
  (`X`, `A`, `lr`, `mu`) matching the source's notation [GPSG §3.16.5]. Local
  only — never a module-level or public API name.

# Appendix B — Local conventions (no external authority — GAP-10)

[GPSG] does not cover ML hygiene. These rules are **authored for this program**;
they are tagged as such wherever they appear.

1. **Seeds.** One seed per run, in the config, recorded in the report. Every
   headline number is reported across ≥ 5 seeds or with a bootstrap CI.
2. **Notebooks explore; modules decide.** A number that reaches a report comes
   from a module invoked by a script, never from notebook state.
3. **Configs are data.** No experiment reads a value that is not in its config.
4. **Datasets are addressed by name and checksum**, never by path.
5. **Every run writes an audit pack** — `config.json`, `report.json`,
   `report.md`, `env.txt`, `dataset_card.json`, plots.
6. **Failures are results.** A model losing to the naive baseline gets reported
   with the same prominence as a win.
7. **A number without a tag does not exist.**

# Appendix C — Experiment report schema

```json
{
  "experiment": "exp04_forecast",
  "run_id": "20260118T0930-exp04_forecast",
  "status": "VERIFIED | PENDING REVISION | UNVERIFIED | BUG",
  "seed": 20260101,
  "dataset_card": {"name": "...", "checksum": "...", "n": 0, "split": "..."},
  "splitter": {"kind": "rolling_window", "folds": 5, "boundaries": []},
  "metric": {"primary": "mase", "secondary": ["rmsse", "rmse"]},
  "arms": [
    {"name": "seasonal_naive", "params": {}, "metrics": {}, "ci": {}, "status": "VERIFIED"}
  ],
  "gates": [{"name": "no_lookforward", "passed": true, "anchor": "TSP p.161"}],
  "gaps": ["GAP-12: runtime unverified on this hardware"],
  "env": {"python": "", "packages": {}, "platform": ""}
}
```

# Appendix D — Reading-order quick reference

| When you need… | Go to |
|---|---|
| The maths behind a method | [MML] — linear algebra p.3–p.262, calculus p.267–p.428, multivariable p.435–p.484, probability p.491–p.629 |
| Scikit-learn mechanics and evaluation discipline | [SKL] ch.3, ch.4, ch.6, ch.9, ch.10 |
| PyTorch mechanics | [PTG] ch.1–ch.3 for fundamentals, ch.10–ch.13 for production |
| Anything temporal, or any question about honest evaluation | [TSP] ch.2–ch.3, then ch.7 for uncertainty |
| How to write the code | [GPSG] p.20 (length), p.27 (docstrings), p.43 (naming), p.46 (types) |
| A paper implemented end to end | `papers/01..30/` — and reproduce the **paper's** number |

---

*Program version 1.0. Every anchor is verifiable in [SOURCE_LEDGER.md](SOURCE_LEDGER.md).
Every result is tagged in [VERIFICATION.md](VERIFICATION.md). Start at Part 1.1.*
