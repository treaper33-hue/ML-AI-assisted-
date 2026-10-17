# Machine Learning — The Complete Program

**From first principles to full professional capability, in a form that can
be checked.**

*Assembled 2026-10-17 · Program version 1.0*

---

## What this is

A 24-week machine-learning curriculum built from five source texts and one
rule: **nothing counts as learned until a numeric gate passes and the result
carries a status tag.**

It is four books and a working testbed. The books tell you what to learn, in
what order, from which pages, and how to prove you learned it. The testbed is
the laboratory you prove it in — one data loader, one metrics module, one set
of splitters, and experiments that declare their acceptance thresholds before
the code runs.

| | |
|---|---|
| **Sources** | *Mathematics of Machine Learning* (Danka, Packt 2025) · *Prompting Scikit-Learn for Machine Learning* · *PyTorch: The Practical Guide* (Gollnick, Rheinwerk 2026) · *Time Series with PyTorch* (Davidson & Ma, Packt 2026) · *Google Python Style Guide* · the 30 paper implementations in `papers/01..30` |
| **Scope** | 6 stages, 30+ modules, 14 competencies, 1 capstone |
| **Effort** | 24 weeks at 10–12 h/week, or 14 weeks at ~20 h/week |
| **Code** | Python, held to the Google Python Style Guide and checked by `ruff` and `mypy` |
| **Testbed** | `ml-mastery/testbed/` — boots with one command |

## How to read it

1. **Boot the testbed first** (Book One, Part 1.1). A program you cannot run
   is a reading list, and reading lists do not produce capability.
2. **Work Book One in order.** Each module names its anchor pages, what to
   build, what to train, and the gate that advances you.
3. **Check every claim against Book Two.** If the ledger has no page anchor
   for something, it does not belong in the program.
4. **Tag every result using Book Three.** An untagged number does not exist.
5. **Keep Book Four beside the keyboard.** It holds the commands.

## Contents

- **[BOOK ONE — The Program](#book-one--the-program)** — The curriculum itself: six stages, thirty-plus modules, each with what to build, what to train, what to execute, the numeric gate that advances you, and the blind spots to watch.
- **[BOOK TWO — The Source Ledger](#book-two--the-source-ledger)** — Where every claim comes from: the algorithms, models and test parameters extracted from the five source texts with page numbers, plus fifteen named gaps.
- **[BOOK THREE — Verification](#book-three--verification)** — The independent review stage: status tags, the eight-step audit, the status board, and the bug log.
- **[BOOK FOUR — Testbed Operations](#book-four--testbed-operations)** — How to boot and drive the runnable half: the one-button setup, the daily commands, the two rules, and the dashboard.

Each book is also a standalone file in `ml-mastery/`: `ML_LEARNING_PROGRAM.md`,
`SOURCE_LEDGER.md`, `VERIFICATION.md` and `testbed/README.md`. The links inside
these pages point at those files and keep working.

## The six guidelines this program was built to satisfy

| Guideline | How it is met |
|---|---|
| Extract algorithms, models and test parameters **with page numbers**; highlight gaps | Book Two: 100+ page-anchored rows, 15 named gaps |
| Assign methodologies to **isolated agents**, unified by a **shared input field** and **single measurement tools** | Book One Part 2: four roles; `mlkit.data` is the only loader, `mlkit.metrics` the only scorer |
| An **independent review stage**; reviewers record bugs; unverified results carry **warning tags** | Book Three: the Auditor, the bug log, four status tags |
| **Controls** to tweak parameters, compare methodologies and read the source | Book One Part 3 and Book Four: TOML configs, `--set`, `--sweep`, `--compare`, `--show-source`, and a read-only dashboard |
| Start with **one document and one baseline experiment**, scale gradually | This document, plus EXP-01 |
| **Pack dependencies and clean data** so an outsider boots it from scratch with one button | `setup.ps1` / `setup.sh`, pinned requirements, generated data with fixed seeds |

## Status at assembly

| Item | State |
|---|---|
| **EXP-01, all four arms** | **VERIFIED** — closed form, hand-written gradient descent, scikit-learn and PyTorch all agree with the analytic optimum (6.2e-15, 2.0e-14, and 0.0198 against a float32 gate of 0.05); test RMSE sits at the irreducible error floor |
| **Independent replay** | **VERIFIED** — a second session re-ran EXP-01 from the same config and seed and reproduced all five gate values |
| Contract tests | passing — metrics, splits and leakage; the 11 Stage 1 gates skip by design, and a skip is not a pass |
| Code contract | `ruff check`, `ruff format --check` and `mypy` clean |
| Environment | conda env `qwen35`, Python 3.11.16, torch 2.13.0 with CUDA; pins not enforced there (GAP-14) |
| Stages 1–5 | specified here, not yet worked — that is the point of the program |

---


# BOOK ONE — The Program

*The curriculum itself: six stages, thirty-plus modules, each with what to build, what to train, what to execute, the numeric gate that advances you, and the blind spots to watch.*

---

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

### How this document implements the Prompt Guidelines

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

## Part 0 — The Outcome Contract

"Full professional capability" is not a feeling. It is this list, each line
provable by an artifact in `runs/`.

### 0.1 Competency matrix

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

### 0.2 Exit criteria

You are done when all of the following are true:

1. All 14 competencies have a `VERIFIED` row on the status board.
2. The capstone runs end to end from a clean clone with one command.
3. An independent reader — a person who has not seen your work — reruns the
   capstone and reproduces every headline number within its stated CI.
4. Every remaining gap is **named in writing**, not silently skipped.

---

## Part 1 — Before Anything Else

### 1.1 Boot the testbed (the one button)

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

### 1.2 Read the baseline experiment before writing any code

`testbed/experiments/exp01_baseline.py` is deliberately small: fit the same
linear model three ways — closed-form normal equations, hand-written gradient
descent, and (when installed) scikit-learn and PyTorch — then score all arms with
**one** metrics module and compare them against the analytic optimum.

It is the template for every later experiment: same config object, same loader,
same scorer, same status tags, same report file.

---

## Part 2 — Operating Model

### 2.1 The four roles (Guideline 2)

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

### 2.2 The shared input field

**`mlkit.data.load_dataset(name, seed)` is the only way data enters an
experiment.** Not `pd.read_csv` in a notebook, not a path hard-coded in a script.
One loader means every methodology is compared on byte-identical inputs — the
precondition for the comparison tables in §3.

Every dataset is either generated from a fixed seed or shipped with a checksum.
The loader returns data plus a `DatasetCard` recording name, shape, seed,
checksum, split policy and provenance. The card goes into the run report.

### 2.3 The single measurement tools

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

### 2.4 Status tags (Guideline 3)

Every result carries exactly one:

| Tag | Meaning |
|---|---|
| **VERIFIED** | Reproduced by the Auditor from a clean run, numeric gate passed. |
| **PENDING REVISION** | A gate failed or a bug is open. The number stays visible; it may not be cited. |
| **UNVERIFIED** | Ran, but has **no numeric gate** — a visual result, a missing dependency, a competency outside the source stack. Must be labelled wherever it appears. |
| **BUG** | A defect with a bug-log ID. Blocks the stage gate. |

An untagged number does not exist. If a plot appears in a report without the tag
of the number behind it, the Interface role has failed.

### 2.5 The loop

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

## Part 3 — Controls (Guideline 4)

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

## Part 4 — The Curriculum

Each module states: **Objective · Anchors · Build · Train · Execute · Gate ·
Controls · Blind spots.** The Gate is the only thing that advances you.

---

### Stage 0 — Testbed and Craft
*Week 0 · about 10 h · Competencies C-14, C-05 (foundation)*

#### M0.1 — Boot and verify the environment
**Objective.** A stranger can reproduce you.
**Build.** Run `setup.ps1`; read what it does line by line.
**Gate.** `python -m mlkit.verify --env` prints interpreter, versions and
platform, and `EXP-01` reports `VERIFIED` on the NumPy arm.
**Blind spots.** GAP-13 — the system interpreter has only NumPy and matplotlib;
every arm needing scikit-learn or PyTorch is `UNVERIFIED` until the venv exists.
GAP-12 — no runtime baselines exist in any source, so record your own.

#### M0.2 — The code contract
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

#### M0.3 — The measurement contract
**Objective.** One definition per metric, forever.
**Anchors.** [TSP p.138] AE/MAE · [p.139] APE · [p.143] the unit-change table ·
[p.144] APE at zeros · [p.150] MASE/RMSSE · [SKL ch.6 "Task-Appropriate Metrics"].
**Build.** Read `mlkit/metrics.py`; add one metric yourself (sMAPE) with its
edge-case test.
**Gate.** `pytest tests/test_metrics.py` green from the testbed directory,
including the zero-division and unit-change tests.
**Blind spots.** GAP-09 — if you ever compute a metric outside this module, every
cross-methodology comparison in the program becomes meaningless.

#### M0.4 — The leakage contract
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

#### M0.5 — Run discipline
**Objective.** Every result is recoverable from its directory alone.
**Anchors.** [SKL ch.14 "What Makes Runs Reproducible", "What to Include in an
Audit Pack"] · [PTG p.363 TensorBoard, p.372 MLflow].
**Gate.** Delete your working directory's `__pycache__`, rerun a `runs/` entry
from its own `config.json`, and reproduce the metrics bit for bit at a fixed seed.

**STAGE 0 GATE:** EXP-01 `VERIFIED` · lint and types clean · metric and leak tests
green · one reproduced run.

---

### Stage 1 — The Mathematical Engine
*Weeks 1–4 · about 45 h · Competencies C-01, C-02, C-03*

Rule for this stage: **implement before you use**. Every routine here exists in
NumPy already; writing it yourself is the point, and the NumPy version is your
test oracle.

#### M1.1 — Vectors, norms, inner products
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

#### M1.2 — Linear maps, determinants, solving systems
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

#### M1.3 — Eigenstructure and factorisations
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

#### M1.4 — Calculus and one-dimensional optimisation
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

#### M1.5 — Gradients in many variables
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

#### M1.6 — Probability, information, likelihood
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

### Stage 2 — Classical ML with Honest Evaluation
*Weeks 5–8 · about 45 h · Competencies C-04, C-05*

#### M2.1 — Framing and schema
**Anchors.** [SKL ch.1 "The Main Types of ML Problems", "Choosing Metrics and
Thresholds That Match with the Goal", "Error Analysis and Slice Checks"] ·
[ch.3 "Foundations: ... Read Data as a Schema", "Data Types on Disk at Load Time"].
**Build.** A framing card and a `schema.yaml` per dataset; a Parquet round-trip.
**Gate.** The metric is chosen and written down **before the first fit**, with the
cost of being wrong stated in one sentence.
**Blind spots.** The most expensive ML mistakes are framing mistakes, and no gate
catches them. This is where a human reviewer is irreplaceable.

#### M2.2 — Pipelines that cannot leak
**Anchors.** [SKL ch.3 "The Preprocessing Toolkit", "Putting It Together: One
Leak-Proof ColumnTransformer", "End-to-End Pipelines"] · [TSP p.158–p.164].
**Build.** One `ColumnTransformer` + `Pipeline` covering imputation, encoding,
scaling, dates and text.
**Train.** Score the pipeline under CV; score the bare model with preprocessing
done up front; report the gap.
**Gate.** The gap is positive and explained in writing.
**Blind spots.** Group and time leakage survive a correct `ColumnTransformer`
[TSP p.161, p.163] — the splitter is a separate defence.

#### M2.3 — Supervised models
**Anchors.** [SKL ch.4 "Linear and Generalized Linear Models", "Trees and
Ensembles", "Ideal Use-Cases for Each Model"] · ridge ties back to [MML p.210].
**Build.** Ridge from scratch (normal equations with a penalty), then logistic
regression, trees, random forest, gradient boosting through sklearn.
**Gate.** Your ridge matches sklearn's coefficients to 1e-6; every model reports a
learning curve, not a single score.
**Blind spots.** A random forest's defaults beat a badly tuned boosted model, and
that comparison teaches nothing — compare at matched budgets.

#### M2.4 — Evaluation, calibration, intervals
**Anchors.** [SKL ch.6] all of: "Task-Appropriate Metrics", "Cross-Validation
Patterns", "Nested CV", "Learning Curves and Validation Curves", "Uncertainty and
Robustness", "Probability Calibration" · [TSP p.136–p.156] for the metric algebra.
**Build.** `nested_cv`, `bootstrap_ci`, reliability diagrams.
**Train.** Flat CV vs nested CV on the same search space.
**Gate.** Nested ≤ flat, with the optimism gap reported; every headline number
carries a 95% CI; Brier score improves after calibration or you explain why not.
**Blind spots.** PR-AUC and ROC-AUC rank models differently under class imbalance
[SKL ch.6] — report both and say which decides.

#### M2.5 — Feature engineering and selection
**Anchors.** [SKL ch.9 "Constructing Features That Generalize", "Datetime:
Snapshot Rules, Lags, and Rollups (without leakage)", "Selecting Features", the
two case studies] · lag mechanics cross-check [TSP p.161].
**Gate.** Selection happens **inside** the CV loop; prove the inflation when it
does not by running it both ways.
**Blind spots.** Feature selection outside the fold is the single most common
silent leak in tabular ML.

#### M2.6 — Tuning
**Anchors.** [SKL ch.10 "Leak-Safe Search over the Whole Pipeline", "Search
Strategies and Space Design", "Hyperparameter Glossary", "Time-Aware Search"].
**Build.** `search_space.yaml`; random search, then a coarse-to-fine refinement.
**Gate.** The search runs over the pipeline; the seed is fixed; the test set is
untouched until the end.
**Blind spots.** More search on a small validation set buys overfitting, not
performance — watch the gap between selection score and test score.

#### M2.7 — Unsupervised structure
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

### Stage 3 — Deep Learning in PyTorch
*Weeks 9–13 · about 55 h · Competencies C-06, C-07*

#### M3.1 — Tensors, autograd, the computation graph
**Anchors.** [PTG p.42] tensors · [p.43] creation and attributes · [p.45] **the
calculation graph and training** · [p.35–p.37] activations · [p.38–p.40] losses ·
[p.40] optimizers · cross-check [TSP ch.4 p.168].
**Build.** ReLU/tanh/sigmoid/softmax and MSE/BCE/CE by hand with their gradients.
**Gate.** Hand gradients match `torch.autograd` to 1e-6 on random inputs; your SGD
step matches `torch.optim.SGD` exactly.
**Blind spots.** Softmax overflows without the max-subtraction trick; [PTG p.37]
does not dwell on it — add the stability test yourself and tag it as locally
authored.

#### M3.2 — The training loop, by hand
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

#### M3.3 — Classification and diagnostics
**Anchors.** [PTG p.93] types · [p.95] confusion matrix · [p.97] ROC · [p.99]
binary · [p.112] multiclass · scoring via `mlkit.metrics` only.
**Gate.** Your by-hand confusion matrix matches `mlkit.metrics`; PR-AUC and
ROC-AUC both reported with bootstrap CIs; macro and micro F1 separated.
**Blind spots.** Accuracy on imbalanced data is a trap already set in Stage 2 —
the deep model does not change the statistics.

#### M3.4 — Convolutional vision
**Anchors.** [PTG p.128] how models handle images · [p.129] architecture ·
[p.134] image classification · [p.163] object detection · [p.178] segmentation ·
[p.188] style transfer.
**Build.** Convolution by hand (im2col), then `F.conv2d`; IoU by hand.
**Gate.** Your convolution matches `F.conv2d` to 1e-5; IoU correct on known boxes;
a top-1 score with a CI.
**Blind spots.** Style transfer [p.188] has **no numeric gate** — it is permanently
`UNVERIFIED` in this program (ledger P-20). Enjoy it; do not cite it.

#### M3.5 — Representation models
**Anchors.** [PTG p.202] recommenders · [p.220] autoencoders · [p.230, p.231]
**VAE** · [p.241–p.250] graph neural networks.
**Gate.** AE reconstruction beats a PCA baseline at equal latent dimension, or you
explain why it does not; the VAE's KL term matches the definition at [MML p.606].
**Blind spots.** A VAE whose KL collapses to zero has become an autoencoder —
monitor both ELBO terms separately, never just their sum.

#### M3.6 — Implement papers from the original text
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

### Stage 4 — Sequences, Uncertainty, and Scale
*Weeks 14–18 · about 55 h · Competencies C-08, C-09, C-10*

This is the hardest stage and the one that most distinguishes a professional:
temporal data punishes every evaluation sin instantly.

#### M4.1 — Temporal diagnostics
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

#### M4.2 — Temporal evaluation
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

#### M4.3 — Neural forecasting baselines
**Anchors.** [TSP ch.5 p.202] simple architectures · [PTG p.261] modelling
approaches · [p.266] a custom model · [p.280] PyTorch Forecasting.
**Gate.** Your MLP forecaster beats seasonal-naive on MASE — **or the failure is
reported as the result**. A deep model that loses to naive is a finding, not an
embarrassment; hiding it is the only failure available here.
**Blind spots.** On short series, naive and ETS frequently win. This is the most
commonly suppressed result in applied forecasting.

#### M4.4 — Optimisation craft
**Anchors.** [TSP ch.6 p.245] LR schedules, regularisation, activations, HPO ·
[PTG p.72] batches · [MML p.481] the underlying method.
**Train.** LR-range test; schedule comparison at a fixed budget.
**Gate.** A loss-vs-LR curve exists and your chosen LR is justified by it.
**Blind spots.** Tuning on the test set through repeated peeking is invisible and
fatal; the schedule choice belongs to validation only.

#### M4.5 — Conformal prediction
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

#### M4.6 — Recurrent and attention architectures
**Anchors.** [TSP ch.8 p.351] RNN/LSTM/GRU, dynamical-systems view · [ch.9 p.388]
transformers · [PTG p.320] the deep dive · [30u30 days 1–3] RNN, LSTM,
regularisation · [days 13–16] attention and transformers.
**Gate.** Your from-scratch LSTM's gradients match `nn.LSTM` to 1e-5; RNN and
transformer forecasters are benchmarked on **identical splits with identical
metrics**.
**Blind spots.** Transformers do not automatically beat an LSTM on small series;
the comparison is only meaningful at matched parameter and compute budgets.

#### M4.7 — Global models and generative sequences
**Anchors.** [TSP ch.10 p.423] other structures (GNN, KAN, foundation models) ·
[ch.11 p.461] transfer learning and global models · [ch.12 p.514] synthetic data
and TimeVAE · [ch.13 p.555] diffusion.
**Gate.** Global model vs per-series models on pooled MASE, same folds.
**Blind spots.** Global models help on many short series and hurt on few long
heterogeneous ones — state which regime you are in before you conclude anything.

#### M4.8 — Classification, clustering, embeddings, anomalies, self-supervision
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

### Stage 5 — Production and Professional Practice
*Weeks 19–22 · about 40 h · Competencies C-11, C-12, C-13*

#### M5.1 — Transfer learning and fine-tuning
**Anchors.** [PTG p.329] Hugging Face · [p.332] transfer learning · [p.335]
fine-tuning a vision model · [p.343] fine-tuning a language model.
**Gate.** Frozen vs fine-tuned at an equal compute budget, with the delta and the
compute cost both reported.

#### M5.2 — Training infrastructure
**Anchors.** [PTG p.351] PyTorch vs Lightning · [p.352] model training · [p.359]
callbacks.
**Gate.** A ported model produces **identical metrics** to your raw loop at the
same seed. If it does not, you have found a real bug — log it.
**Blind spots.** GAP-05 — nothing here covers DDP/FSDP; multi-GPU capability stays
`UNVERIFIED`.

#### M5.3 — Tracking and monitoring
**Anchors.** [PTG p.363] TensorBoard · [p.372] MLflow · [p.377] Weights & Biases ·
[SKL ch.14 "What Makes Runs Reproducible"].
**Gate.** Delete a run's local artifacts and rebuild the result from the tracking
record alone.

#### M5.4 — Deployment
**Anchors.** [PTG p.385] strategies · [p.387] local deployment · [p.393] Heroku ·
[p.399] Azure.
**Build.** A local inference service with input validation and a health check.
**Gate.** p50 and p95 latency under a fixed load, plus a documented rollback path.
**Blind spots.** GAP-07 — no containers, CI/CD, feature stores or registries in
these sources. This module is an **introduction**; the competency stays
`UNVERIFIED` until closed externally.

#### M5.5 — Explainability
**Anchors.** [SKL ch.11 "Global Importance That Holds up: Permutation Importance",
"Directionality and Effect Size: Partial Dependence and ICE", "Interpretable
Baselines", "Slice Analysis", "Local 'Why This Prediction?'"].
**Gate.** Permutation importance computed on held-out data; PDP and ICE plotted
together (PDP alone hides heterogeneity); one slice where the model fails, named.

#### M5.6 — Responsible AI and governance
**Anchors.** [SKL ch.14] scope and stakeholders, data governance, fairness and
harm, transparency, integrity and robustness, human-in-the-loop, monitoring and
incidents, reproducibility and the audit pack · [ch.15], [ch.16].
**Gate.** A model card, a fairness/harm assessment with a named protected
attribute, an incident plan with a rollback trigger, and an audit pack a stranger
can rerun.

#### M5.7 — Frontier reading
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

## Part 5 — Capstone
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

## Part 6 — Schedules

### 6.1 Standard track — 24 weeks at 10–12 h/week

| Weeks | Stage | Headline deliverable |
|---|---|---|
| 0 | Testbed and Craft | EXP-01 verified, contracts in place |
| 1–4 | Mathematical Engine | `mlkit` math modules, LR divergence study |
| 5–8 | Classical ML | EXP-02 leak-proof tabular benchmark |
| 9–13 | Deep Learning | hand-written loop + one paper reproduced |
| 14–18 | Sequences and Uncertainty | EXP-04 forecasting with conformal coverage |
| 19–22 | Production | tracked, served, explained, governed model |
| 23–24 | Capstone | reproduced by an independent reader |

### 6.2 Compressed track — 14 weeks at about 20 h/week

Weeks 1–2 Stage 0 + Stage 1 (M1.1–M1.3, M1.5, M1.6; skim M1.4's theory but keep
the LR study) · 3–5 Stage 2 (keep M2.2, M2.4; compress M2.7) · 6–8 Stage 3 (keep
M3.1, M3.2, M3.6; M3.4 light) · 9–12 Stage 4 (no cuts — this stage is where the
professional distinction lives) · 13 Stage 5 (M5.3, M5.4, M5.6) · 14 Capstone.

**Never compressible:** the metric contract (M0.3), the leakage contract (M0.4),
temporal evaluation (M4.2), and conformal coverage (M4.5). Everything else can be
re-learned later; these four are the ones that make every other number true or
false.

### 6.3 The daily shape

| Minutes | Role | Activity |
|---|---|---|
| 30 | Analyst | read the anchored pages, write the ledger row and the threshold |
| 60 | Coder | implement, test, lint |
| 20 | Auditor | run the gate, tag the result, log bugs |
| 10 | Interface | update the board, commit the run |

---

## Appendix A — The style contract (enforced)

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

## Appendix B — Local conventions (no external authority — GAP-10)

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

## Appendix C — Experiment report schema

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

## Appendix D — Reading-order quick reference

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

---


# BOOK TWO — The Source Ledger

*Where every claim comes from: the algorithms, models and test parameters extracted from the five source texts with page numbers, plus fifteen named gaps.*

---

> **Step 1 of the Prompt Guidelines:** *"Extract working algorithms, mathematical
> models, and test parameters. Be sure to specify page numbers and highlight gaps."*

This ledger is the single source of truth for *where* every claim in
[ML_LEARNING_PROGRAM.md](ML_LEARNING_PROGRAM.md) comes from. Nothing enters the
program without an anchor here. Nothing leaves the review stage without a status
tag.

---

### 0. Pagination basis (read this before citing anything)

| Tag | Source | Edition / ID | Citation basis | Reliability |
|---|---|---|---|---|
| **[MML]** | *Mathematics of Machine Learning* — Tivadar Danka, Packt, May 2025 | ISBN 978-1-83702-787-3 | **Printed page numbers** from the book's own ToC | High — ToC and body pagination agree |
| **[SKL]** | *Prompting Scikit-Learn for Machine Learning* (EPUB, 16 chapters) | EPUB, no page map | **Chapter + section heading** | Medium — see GAP-01 |
| **[PTG]** | *PyTorch: The Practical Guide* — Bert Gollnick, Rheinwerk, 2026 | ISBN 978-1-4932-2786-0 | **Printed page numbers** from the book's own ToC | High |
| **[TSP]** | *Time Series with PyTorch* — Graeme Davidson & Lei Ma, Packt, May 2026 | ISBN 978-1-80512-818-2 | **PDF page index** (836-page file, reflowed) | Medium — see GAP-02 |
| **[GPSG]** | *Google Python Style Guide* (PDF, 56 pp.) | — | **PDF page index** | High |
| **[30u30]** | `papers/01..30/` in this repository | local clone | **Directory + file** | Low — see GAP-11 |

**Citation format used throughout:** `[MML p.207]`, `[TSP p.158]`, `[PTG p.76]`,
`[GPSG p.43]`, `[SKL ch.6 "Nested CV"]`, `[30u30 day 13]`.

---

### 1. [MML] Mathematics of Machine Learning — algorithm extraction

#### Part I: Linear Algebra (p.1–262)

| # | Algorithm / model | Anchor | You must implement | Test parameter / acceptance |
|---|---|---|---|---|
| M-01 | Vector spaces, basis, span, subspace | p.9, p.14, p.18, p.23 | `is_independent(vectors)` via rank | rank of a known-rank 5x3 design matrix |
| M-02 | NumPy arrays as vectors; vectorisation speedup | p.29, p.35, **p.37 "Is NumPy really faster than Python?"** | timed dot product, pure Python vs NumPy | speedup ratio at n = 1e6; **no accepted constant — measure on your box** |
| M-03 | Norms and induced distances | p.49, p.55 | `lp_norm(x, p)` for p in {1, 2, inf} | norm2((3,4)) = 5.0 exactly |
| M-04 | Inner products, orthogonality, angles | p.56, p.60, p.61 | `cosine_similarity` | orthogonal basis vectors give 0.0 +/- 1e-12 |
| M-05 | **Gram–Schmidt orthogonalisation** | theory **p.67**, code **p.89** | `gram_schmidt(A)` | QtQ = I within 1e-10; reorthogonalise if cond(A) > 1e8 |
| M-06 | Orthogonal complement | p.71 | projection onto the complement | P^2 = P and P = P^T |
| M-07 | Matrices as linear maps; matrix multiplication revisited | p.94, p.107 | `matmul` by hand, then vs `np.matmul` | agreement to 1e-12 on 64x64 random |
| M-08 | Matrices and data (design-matrix view) | p.110 | load a tabular set as X in R^(n x d) | shape assertions |
| M-09 | Plane transforms: stretch, rotate, shear, reflect, project | p.133–p.138 | 2x2 transform gallery | determinant signs / areas match p.140 |
| M-10 | Determinants and volume scaling | p.139, p.140, p.148 | `det` via LU | matches `np.linalg.det` to 1e-10 |
| M-11 | **Gaussian elimination** + complexity | p.160, **p.165** | `gauss_solve(A, b)` | O(n^3) scaling check at n in {64, 128, 256} |
| M-12 | **LU decomposition** | p.168, code **p.171** | `lu(A)` with partial pivoting | norm(PA - LU) < 1e-10 |
| M-13 | Matrix inversion, done properly | p.167, **p.174, p.177 "How to actually invert matrices"** | solve, never invert | residuals of `inv(A)@b` vs `solve(A,b)` |
| M-14 | Determinants in practice (recursive vs LU) | p.178–p.182 | both, timed | recursion becomes unusable near n = 12 |
| M-15 | Eigenvalues, characteristic polynomial, eigenvectors | p.189, p.192, p.193 | `power_iteration` | dominant eigenpair of a symmetric PSD matrix |
| M-16 | Spectral decomposition (self-adjoint) | p.203 | `eigh`-based reconstruction | norm(A - Q L Qt) < 1e-10 |
| M-17 | **Singular Value Decomposition** | **p.207** | `svd_lowrank(A, k)` | rank-k error equals sqrt(sum of squared tail singular values) |
| M-18 | Orthogonal projections | p.210 | least squares via projection | matches the normal-equation solution |
| M-19 | Computing eigenvalues; **QR algorithm** | p.217, **p.230** | `qr_algorithm(A, iters)` | converges on symmetric A; log iteration count |
| M-20 | Graph/matrix duality, Frobenius normal form | p.243, p.244, p.251 | adjacency to reachability | irreducibility check |

#### Parts II–III: Calculus and Multivariable Calculus (p.265–484)

| # | Algorithm / model | Anchor | You must implement | Test parameter / acceptance |
|---|---|---|---|---|
| M-21 | Sequences, series, convergence | p.287 | convergence tester | geometric vs harmonic series |
| M-22 | Limits, continuity, topology | p.321 | theory gate only | written proof sketch |
| M-23 | Differentiation, numerically | p.343 | central-difference derivative | error ~ O(h^2); find the h that minimises total error |
| M-24 | Minima/maxima via derivatives | p.372 | critical-point finder | on f(x) = x^4 - 3x^2 + 1 |
| M-25 | **Gradient descent (1-D)** | **p.381**, justification **p.389** | `gd(f, df, x0, lr, iters)` | converges for lr < 2/L; **demonstrate divergence above it** |
| M-26 | Integration: theory and practice | p.408, p.418 | trapezoid + Simpson | integral of x^2 on [0,1] = 1/3 +/- 1e-9 |
| M-27 | Curse of dimensionality | **p.443** | volume-of-shell experiment | mass near the surface vs d in {2..100} |
| M-28 | Partial and total derivatives; Jacobians | p.448, **p.460** | numeric `jacobian(f, x)` | vs analytic Jacobian of a 3→2 map |
| M-29 | **Gradient descent, full multivariable form** | **p.481** | `gd_nd` with a momentum toggle | Rosenbrock: iterations to grad-norm < 1e-4 |

#### Part IV: Probability (p.489–629)

| # | Algorithm / model | Anchor | You must implement | Test parameter / acceptance |
|---|---|---|---|---|
| M-30 | Axioms of probability | p.498 | theory gate only | Kolmogorov axioms restated |
| M-31 | Conditional probability, Bayes | p.515 | `bayes_update` | base-rate problem; posterior to 1e-12 |
| M-32 | Random variables; discrete distributions | p.534, p.542 | Bernoulli / Binomial / Poisson samplers | chi-squared goodness of fit, alpha = 0.05 |
| M-33 | Real-valued distributions and densities | p.556, p.567 | Gaussian pdf/cdf | vs `scipy.stats` to 1e-12 |
| M-34 | Expected value and its properties | p.580, p.585, p.587 | Monte-Carlo E[X] | CI covers the analytic value in ~95% of 200 repeats |
| M-35 | Variance | p.589 | Welford online variance | matches two-pass to 1e-12 |
| M-36 | **Law of large numbers** | **p.595** | convergence plot | sample-mean error shrinks as O(n^-1/2) |
| M-37 | **Information theory: entropy, cross-entropy, KL** | **p.606** | `entropy`, `kl_divergence` | KL(p,p) = 0; KL >= 0 on 1000 random pairs |
| M-38 | **Maximum Likelihood Estimation** | **p.619** | MLE for Gaussian and Bernoulli | recovers mu, sigma^2 from 1e5 samples within 3 SE |

**Appendices:** logic **p.635**, structure of mathematics **p.647**, set theory
**p.661**, complex numbers **p.673** — reference material, not a reading block.

---

### 2. [SKL] Prompting Scikit-Learn — workflow extraction

Anchors are chapter + section heading (see GAP-01).

| # | Model / procedure | Anchor | You must implement | Acceptance |
|---|---|---|---|---|
| S-01 | Problem types; features, target, splits | ch.1 "The Main Types of ML Problems", "Data, Features, and Splits" | a problem-framing card per project | a reviewer can restate the target without reading code |
| S-02 | Metric and threshold tied to the cost of being wrong | ch.1 "Choosing Metrics and Thresholds That Match with the Goal" | `decide_metric()` note in each config | metric chosen **before** the first fit |
| S-03 | Uncertainty and abstention | ch.1 "Uncertainty and Abstention" | coverage-vs-accuracy curve | connects to conformal prediction [TSP p.309] |
| S-04 | Reading data as a schema | ch.3 "Foundations: Why Preprocessing Matters and How to Read Data as a Schema" | `schema.yaml` per dataset | every column typed before modelling |
| S-05 | Storage formats: CSV, Parquet, ORC, JSONL, XLSX | ch.3 "Data Types on Disk at Load Time" | Parquet round-trip | dtype-stable round-trip |
| S-06 | **Leak-safe splits: stratified, group, time-aware** | ch.3 "Leakage-Safe Evaluation: Choosing Splits" | `make_splitter(kind)` | cross-checked against [TSP p.158–164] |
| S-07 | Imputation, encoding, scaling, dates, text | ch.3 "The Preprocessing Toolkit" | one `ColumnTransformer` | fitted on train folds only |
| S-08 | **One leak-proof `ColumnTransformer` + `Pipeline`** | ch.3 "Putting It Together: One Leak-Proof ColumnTransformer", "End-to-End Pipelines" | the whole estimator as one object | show the gap between scoring the pipeline and scoring the bare model |
| S-09 | Ridge, Lasso, Logistic regression; regularisation | ch.4 "Linear and Generalized Linear Models" | ridge from scratch vs sklearn | coefficients agree to 1e-6 |
| S-10 | Trees, Random Forests, Gradient Boosting | ch.4 "Trees and Ensembles" | depth and leaf sweeps | learning curves, not single scores |
| S-11 | Thresholds, calibration, slice analysis | ch.4 "Evaluation That Holds up"; ch.6 "Probability Calibration" | reliability diagram | Brier score before and after calibration |
| S-12 | Clustering: KMeans, Agglomerative, DBSCAN, Spectral | ch.5 "Clustering Algorithms" | parameter sweep grid | silhouette plus stability across seeds |
| S-13 | PCA, TruncatedSVD, KernelPCA; t-SNE/UMAP as **visualisation only** | ch.5 "Dimensionality Reduction" | PCA from SVD by hand | matches sklearn PCA up to sign; ties to [MML p.207] |
| S-14 | Cluster evaluation: elbow, silhouette, Davies–Bouldin, stability | ch.5 "Evaluation and Diagnostics" | `cluster_report()` | reported with seed variance |
| S-15 | Classification, regression and ranking metrics | ch.6 "Task-Appropriate Metrics" | they live in `mlkit/metrics.py` **only** | one implementation, many callers |
| S-16 | **CV patterns: k-fold, group, time-aware, nested** | ch.6 "Cross-Validation Patterns", "Nested CV" | `nested_cv()` | nested score <= flat score; report the optimism gap |
| S-17 | Learning and validation curves | ch.6 "Learning Curves and Validation Curves" | both | bias vs variance diagnosed in writing |
| S-18 | Bootstrap CIs; seed and slice stability | ch.6 "Uncertainty and Robustness" | `bootstrap_ci(metric, B=1000)` | every headline number carries a CI |
| S-19 | Feature construction: datetime lags and rollups, TF-IDF to SVD, interactions | ch.9 "Constructing Features That Generalize" | leakage-free lag builder | lag features built inside the fold |
| S-20 | Selection: filter, wrapper, embedded | ch.9 "Selecting Features" | all three, compared | selection inside the CV loop only |
| S-21 | Search strategies and space design | ch.10 "Search Strategies and Space Design", "Hyperparameter Glossary" | `search_space.yaml` | search over the **pipeline**, not the estimator |
| S-22 | Explainability: permutation importance, PDP/ICE, standardised coefficients, local explanations | ch.11 "Explainability Toolbox" | `explain()` report | permutation importance on held-out data only |
| S-23 | Responsible AI: governance, fairness, robustness, human-in-the-loop, monitoring | ch.14 (all sections) | model card + incident plan | — |
| S-24 | **Reproducibility and the audit pack** | ch.14 "What Makes Runs Reproducible", "What to Include in an Audit Pack" | `artifacts/` per run | a stranger reruns it and matches |

---

### 3. [PTG] PyTorch: The Practical Guide — implementation extraction

| # | Topic | Anchor (printed pages) | You must implement | Acceptance |
|---|---|---|---|---|
| P-01 | Activations: ReLU, tanh, sigmoid, softmax | **p.35–p.37** | each by hand, then autograd check | gradient vs `torch.autograd` to 1e-6 |
| P-02 | Losses: regression, binary, multiclass | **p.38–p.40** | MSE, BCE, CE by hand | vs `torch.nn` to 1e-6 |
| P-03 | Optimizers and the update rule | **p.40–p.41** | SGD by hand | matches `torch.optim.SGD` step for step |
| P-04 | Tensors: creation and attributes | p.42, p.43 | tensor drill | dtype / device / shape assertions |
| P-05 | **Computation graph and training** | **p.45–p.50** | manual backward trace | hand-computed gradients vs autograd |
| P-06 | Data prep: feature types, one-hot, EDA, scaling | p.52, p.54, p.56, p.59 | preprocessing function | scaler fitted on train only |
| P-07 | First model: import, train, evaluate, infer | p.61, p.62, p.64, p.65 | the full loop, no abstractions | loss decreases monotonically at a fixed seed |
| P-08 | Model class plus optimizer idiom | **p.68** | `nn.Module` subclass | parameter count matches your hand count |
| P-09 | **Batches: why and how** | p.72–p.75 | batch-size sweep | wall clock and final loss for batch in {1, 8, 32, 256} |
| P-10 | **Dataset and DataLoader** | **p.76** | custom `Dataset` | deterministic order at a fixed seed |
| P-11 | Saving and loading a model | **p.80** | checkpoint round-trip | reloaded model reproduces logits bit for bit |
| P-12 | Data sampling, including imbalance | p.83–p.91 | weighted sampler | per-batch class balance within 5% |
| P-13 | Classification types; confusion matrix; ROC | p.93, **p.95**, **p.97** | confusion matrix by hand | matches `mlkit.metrics` |
| P-14 | Binary classification, end to end | **p.99** | full run | PR-AUC and ROC-AUC with bootstrap CIs |
| P-15 | Multiclass classification | **p.112** | full run | macro and micro F1 reported separately |
| P-16 | CNNs: how models handle images; architecture | p.128, p.129 | convolution by hand (im2col) | vs `F.conv2d` to 1e-5 |
| P-17 | Image classification | **p.134** | train on a small set | top-1 with CI, plus confusion matrix |
| P-18 | Object detection | **p.163** | IoU and mAP by hand | IoU on known boxes |
| P-19 | Semantic segmentation | **p.178** | Dice / IoU loss | vs a library implementation |
| P-20 | Style transfer | p.188 | optional, visual only | **no numeric gate — tag UNVERIFIED** |
| P-21 | Recommender systems | p.199, **p.202** | matrix-factorisation model | ties to SVD [MML p.207] |
| P-22 | Autoencoders | p.220 | AE on tabular or image data | reconstruction MSE vs a PCA baseline |
| P-23 | **Variational autoencoders** | **p.230, p.231** | VAE with the ELBO | KL term matches the definition at [MML p.606] |
| P-24 | Graph neural networks | p.241, p.246, **p.250** | message passing | ties to [30u30 day 21] |
| P-25 | Time-series forecasting, custom model | **p.266** | windowed dataset | no lookforward leakage [TSP p.161] |
| P-26 | PyTorch Forecasting library | **p.280** | the same task, library version | metrics match the custom model within noise |
| P-27 | LLM usage: params, templates, chains, structured outputs | p.290, p.304, p.311, p.315, **p.317** | one structured-output call | schema-validated response |
| P-28 | **How transformers work (deep dive)** | **p.320** | attention by hand | vs [30u30 day 13] |
| P-29 | Pretrained models (Hugging Face); transfer learning | p.329, **p.332** | freeze/unfreeze study | frozen vs fine-tuned at the same budget |
| P-30 | Fine-tuning a CV model / a language model | **p.335**, **p.343** | both | report compute and the delta over baseline |
| P-31 | **PyTorch Lightning** | p.351, **p.352**, callbacks **p.359** | port one model | identical metrics to the raw loop |
| P-32 | TensorBoard, MLflow, Weights & Biases | **p.363**, **p.372**, **p.377** | log one run in each | the run is recoverable from the log alone |
| P-33 | Deployment: strategies, local, Heroku, Azure | **p.385**, **p.387**, p.393, p.399 | local service first | p50 / p95 latency under load |

---

### 4. [TSP] Time Series with PyTorch — extraction (PDF page index)

| # | Topic | Anchor (PDF pages) | You must implement | Acceptance |
|---|---|---|---|---|
| T-01 | Time-series framing | ch.1 p.31 | — | vocabulary quiz |
| T-02 | Missing data: types and mechanism | p.52, **p.53** | MCAR / MAR / MNAR argument per column | written justification |
| T-03 | Simple imputation and **MICE** | p.57, **p.60** | both | compare the downstream metric, not imputation error alone |
| T-04 | Patterns: trend, seasonality, cycles | p.64, p.65 | decomposition plot | — |
| T-05 | **Additive vs multiplicative decomposition** | **p.66**, **p.70** | both, with the formulas | residual mean near 0 |
| T-06 | STL, wavelet, TBATS alternatives | **p.73** | STL | seasonal-strength statistic |
| T-07 | Seasonality plotting | p.76 | seasonal subseries plot | — |
| T-08 | Dependence; **ACF and PACF** | p.79, **p.84**, **p.86** | both by hand | vs `statsmodels` to 1e-8 |
| T-09 | Feature significance via OLS and tree importance | p.90, p.91 | both | agreement and disagreement documented |
| T-10 | **Stationarity: ADF, KPSS, PP** | p.92, **p.97** | run all three | **ADF and KPSS can disagree — record both, never cherry-pick** |
| T-11 | Structures: univariate, multivariate, multivariable, panel | p.103–p.106 | classify your dataset | the wrong class means the wrong splitter |
| T-12 | Why we evaluate; validation discipline | p.113, p.116, p.118 | — | — |
| T-13 | **Fixed-origin; train/test; train/val/test** | **p.126**, p.129 | `fixed_origin_split()` | no future in train |
| T-14 | **Expanding vs rolling-window CV** | **p.132**, **p.134** | both | fold boundaries printed and asserted |
| T-15 | Neural-network validation shortcuts, and why | p.136 | — | state your choice explicitly |
| T-16 | **Intrinsic metrics: AE, MAE, MSE, RMSE, APE** | p.137, **p.138**, **p.139**, **p.144** | in `mlkit/metrics.py` | **APE is undefined at zeros (p.144) — test that case** |
| T-17 | Scale dependence (the Celsius vs Kelvin table) | **p.143** | unit-change test | MAPE shifts under a unit change; MASE must not |
| T-18 | **Extrinsic metrics: RelMAE, RelMSE, RelRMSE, RSE, MASE, MdMASE, RMSSE** | **p.149**, **p.150** | MASE and RMSSE | the naive-forecast denominator uses **train only** |
| T-19 | Grouped metrics; multivariate RMSE | p.154, **p.155** | aggregation rules | per-series and pooled, both reported |
| T-20 | Which metric, when | p.155, **p.156** | decision table in the config | large panels mean RMSSE |
| T-21 | Length of evaluation | **p.156** | horizon sweep | metric-vs-horizon curve |
| T-22 | **Leakage: transformation-based, lookforward, panel** | **p.158**, **p.159**, **p.161**, **p.163**, tips **p.164** | leakage test suite | **the canonical leakage reference for every stage** |
| T-23 | PyTorch fundamentals (tensors, autograd) | ch.4 p.168 | cross-check vs [PTG p.42–p.50] | — |
| T-24 | Simple neural architectures from first principles | ch.5 p.202 | MLP forecaster | beats seasonal-naive on MASE, or is reported as a failure |
| T-25 | **Optimization: LR schedules, regularisation, activations, HPO** | ch.6 p.245 | LR-range test | loss-vs-LR curve |
| T-26 | Uncertainty quantification | p.311 | prediction intervals | — |
| T-27 | **Conformal prediction: intuition, split, quantiles** | **p.316**, **p.317**, **p.323**, **p.325** | split conformal | empirical coverage within 2% of 1 - alpha |
| T-28 | Arbitrary confidence levels; the probability-bound caveat | **p.331**, **p.332** | alpha sweep | **the upper bound quoted in the literature may not hold (p.332) — flag it** |
| T-29 | Splitting: bootstrap, cross-conformal, jackknife | **p.333** | one of the three | — |
| T-30 | Conformalised regression and forecasting | **p.334**, **p.335** | both | coverage under distribution shift |
| T-31 | **EnbPI**, including applying it to a PyTorch model | **p.337**, **p.340** | EnbPI wrapper | coverage and interval width |
| T-32 | Evaluating interval quality | **p.346** | coverage and width report | pinball loss or Winkler score |
| T-33 | RNN / LSTM / GRU, dynamical-systems view | ch.8 p.351 | from scratch, then `nn.LSTM` | gradients match; ties to [30u30 days 1–3] |
| T-34 | Transformers for forecasting | ch.9 p.388 | benchmark vs the RNN | same splits, same metrics |
| T-35 | GNNs, Kolmogorov–Arnold Networks, foundation models | ch.10 p.423 | one of the three | — |
| T-36 | Transfer learning and global models | ch.11 p.461 | global vs per-series | pooled MASE |
| T-37 | Synthetic data; TimeVAE | ch.12 p.514 | TimeVAE | distributional similarity test |
| T-38 | Diffusion models for probabilistic forecasting | ch.13 p.555 | forward and reverse process | sample coverage |
| T-39 | TS classification (distance, feature, shapelet, dictionary, convolution, deep) via `aeon` | ch.14 p.586 | two families | accuracy with CI |
| T-40 | TS clustering | ch.15 p.638 | featurise then cluster | stability across seeds |
| T-41 | Embeddings for time series | ch.16 p.676 | embed, then a downstream task | lift over raw features |
| T-42 | Anomaly detection: rules, statistics, isolation forest, autoencoder, Matrix Profile | ch.17 p.690 | two methods | precision@k against labelled anomalies |
| T-43 | Self-supervised learning; contrastive predictive coding | ch.18 p.763 | CPC pretraining | label-efficiency curve |

**Book code bundle:** `https://github.com/PacktPublishing/Time-Series-with-PyTorch`
— reconstructed across a line break in the PDF text layer. **Verify before cloning** (GAP-03).

---

### 5. [GPSG] Google Python Style Guide — the code contract

Every line of code in this program obeys these rules.

| Rule | Anchor | Enforcement |
|---|---|---|
| Lint clean (pylint) | **p.1 §2.1** | lint gate |
| Imports: modules and packages only | **p.2 §2.2** | lint |
| Exceptions: never a bare `except:` | **p.5 §2.4** | lint |
| **Type-annotated code** | **p.19 §2.21**, **p.46 §3.19** | type checker on `mlkit/` |
| **Maximum line length 80** — exceptions are long imports, URLs, pathnames, long flags in comments, long module-level string constants, and pylint-disable comments | **p.20 §3.2** | formatter at 80 |
| Indentation: 4 spaces, never tabs | **p.23 §3.4** | formatter |
| Whitespace conventions | **p.25 §3.6** | formatter |
| **Docstrings** on every public module, function, class and method | **p.27 §3.8, §3.8.1** | lint |
| Function docstrings carry Args / Returns / Raises | **p.29 §3.8.3** | review |
| Class docstrings carry Attributes | **p.33 §3.8.4** | review |
| Strings: f-strings are fine; keep `%`-style lazy formatting for logging | **p.35 §3.10** | review |
| `TODO(username): ...` format | **p.39 §3.12** | lint |
| Import formatting: one per line, grouped stdlib / third-party / local | **p.40 §3.13** | formatter |
| **Naming**: `module_name`, `ClassName`, `function_name`, `GLOBAL_CONSTANT`, `_internal` | **p.43 §3.16, §3.16.2** | lint |
| **`main()` plus `if __name__ == '__main__':`** in every executable | **p.45 §3.17** | review |
| **Function length**: rethink anything past about 40 lines | **p.46 §3.18** | review |
| Mathematical notation may break naming rules when local and matching the source | **§3.16.5** | documented exemption |

---

### 6. Gaps and blind spots requiring additional verification

Per the Prompt Guidelines, these stay visible and tagged until closed.

| ID | Status | Gap | Impact | How to close |
|---|---|---|---|---|
| **GAP-01** | open | **[SKL] is an EPUB with no stable pagination.** All anchors are chapter plus section heading. | Citations cannot be page-verified; another edition may rename sections. | Cite headings verbatim; re-resolve if the file is replaced. |
| **GAP-02** | open | **[TSP] page numbers are PDF page indices** of the 836-page file, not printed-book pages. | They will not match a print copy. | Keep using the supplied PDF, or re-map once a printed edition exists. |
| **GAP-03** | pending | **[TSP] code-bundle URL was reconstructed across a line break.** | A wrong URL sends you to the wrong repository. | Open the link by hand before cloning. |
| **GAP-04** | **hole** | **No reinforcement-learning theory in any of the four books.** PPO and RLHF exist only as [30u30 days 29–30]. | Stage 5 RL content rests on unaudited local code. | Add Sutton & Barto or the original PPO / RLHF papers before claiming RL competence. |
| **GAP-05** | **hole** | **No distributed-training coverage** (DDP, FSDP, ZeRO). Nearest items: [PTG p.351 Lightning], [30u30 day 24 GPipe]. | Capability at scale is unproven. | External source required; do not self-certify this competency. |
| **GAP-06** | **hole** | **No statistical learning theory** — no PAC learning, VC dimension, Rademacher complexity or concentration inequalities. [MML] stops at the law of large numbers (p.595) and information theory (p.606). | You can compute but not bound generalisation. | Optional external reading; flagged rather than faked. |
| **GAP-07** | **hole** | **Production coverage is thin.** [PTG p.385–407] covers local, Heroku and Azure; no containers, CI/CD, feature stores, or registries beyond MLflow (p.372). | The deployment module is demo-grade. | Treat Stage 5 deployment as an introduction; keep that competency tagged UNVERIFIED. |
| **GAP-08** | resolved by rule | **Three sources define leakage differently.** | Inconsistent guardrails. | **Arbitration rule:** [TSP p.158–164] is canonical for leakage, including non-temporal work; [SKL ch.3] supplies the sklearn mechanics. |
| **GAP-09** | resolved by rule | **Metric definitions overlap and can disagree** — APE at zeros [TSP p.144], MAPE under unit change [TSP p.143], macro vs micro F1 [SKL ch.6]. | Two modules could report "RMSE" and mean different things. | **One implementation only**, in `testbed/mlkit/metrics.py`, with the edge cases as tests. |
| **GAP-10** | open | **[GPSG] says nothing about ML-specific hygiene** — seeds, notebook discipline, experiment configs, data versioning. | This program's conventions there have no external authority. | Locally authored in the program's Appendix B and tagged as such. |
| **GAP-11** | pending | **[30u30] implementations are educational and un-audited.** The README asserts all units complete; no reproduction against the papers' reported numbers is shown. | Using them as ground truth imports unknown bugs. | Every 30u30 output stays PENDING REVISION until you reproduce one headline number from the paper. |
| **GAP-12** | open | **No runtime or hardware baselines in the sources.** [TSP] calls a CUDA GPU "beneficial but not required"; [PTG] assumes CPU is fine. | Every time estimate here is an estimate. | Record your own wall-clock in `runs/` and replace the estimates. |
| **GAP-13** | **closed (2026-10-17)** | **Dependency coverage.** The default system interpreter has only `numpy` and `matplotlib`. The conda env **`qwen35`** (Python 3.11.16) now carries scipy, pandas, scikit-learn, matplotlib, pyarrow, statsmodels, pytest, ruff, mypy, and torch 2.13.0 with working CUDA 13.4 on an RTX 5070 Laptop GPU. | None remaining — every stage can run in `qwen35`. | Run `setup.ps1 -CondaEnv qwen35` (installs only what is absent) or `setup.ps1` (clean, pinned `.venv`). A broken package is tagged `BROKEN` by `verify --env` and `BUG` by the arm — never a silent skip. |
| **GAP-15** | open | **No consistent CUDA torchaudio on this stack.** conda-forge ships torch 2.13 for Python 3.11 only as a `cuda134` build, while every conda-forge torchaudio CUDA build is `cuda130`; torchaudio refuses the mismatch at import. Matching `gpu_cuda130` torch builds exist only on anaconda `main`, which stops at 2.9.1 for py311 and is excluded by strict channel priority. | No stage in this program needs torchaudio, so no gate is blocked — but audio work in `qwen35` is not possible as it stands. | Either move torch/torchvision/torchaudio together to pip's `cu130` wheels (self-consistent upstream, several GB, and conda must release its torch packages first), or use a Python version where the conda-forge builds line up. |
| **GAP-14** | open | **Pins are not enforced in an existing environment.** `qwen35` runs numpy 2.4.6, scipy 1.17.1, pandas 3.0.6 and scikit-learn 1.9.1 — all newer than `requirements.txt`. Matching the pins would have downgraded a working environment. | Numbers produced there reproduce on this machine, not from a clean clone. | Record the environment with every run (`env.txt` already does). Use the default `.venv` mode for anything that must reproduce elsewhere. |

---

*Ledger version 1.0 — every row above traces to a file on disk. Rows without a
page anchor say so and carry a gap ID.*

---


# BOOK THREE — Verification

*The independent review stage: status tags, the eight-step audit, the status board, and the bug log.*

---

> **Step 3 of the Prompt Guidelines:** *"Set up an independent review stage: let
> reviewers record bugs, and leave all unverified results clearly marked with
> warning tags."*

This file is maintained by the **Auditor** role only. The Coder may add bug
entries; the Coder may not change a status tag.

---

### 1. Status tags

| Tag | Meaning | May be cited? |
|---|---|---|
| **VERIFIED** | Reproduced from a clean run by the Auditor; the numeric gate passed. | Yes |
| **PENDING REVISION** | A gate failed, or an open bug touches it. | **No** |
| **UNVERIFIED** | Ran, but has no numeric gate — visual output, a missing dependency, or a competency outside the source stack. | Only with the tag attached |
| **BUG** | Defect with a log ID below; blocks the stage gate. | **No** |

**Warning-tag rule.** Any `UNVERIFIED` or `PENDING REVISION` number must carry its
tag in every place it appears — report, plot title, table, and conversation. A
number that travels without its tag is the failure mode this whole structure
exists to prevent.

---

### 2. The review stage

Run this for every artifact before its stage gate. The Auditor works from the run
directory, **not** from the Coder's screen or explanation.

1. **Independence.** Open only `runs/<id>/`. If you cannot tell what was run from
   `config.json` and `report.json` alone, that is finding #1.
2. **Re-derivation.** Recompute the headline metric by a second route — a
   different implementation, a closed-form value, or a hand calculation on a
   small case. Agreement within tolerance, or it fails.
3. **Replay.** Rerun from `config.json` at the recorded seed. Metrics must match
   bit for bit, or the run must explain its own nondeterminism.
4. **Leak hunt.** Walk the [TSP p.164] checklist: split before any transform;
   preprocessing fitted on train folds only; parameterised transforms (scalers,
   imputers, encoders) fitted inside the fold; temporal order respected; panel
   entities not split across train and test.
5. **Metric audit.** Confirm every number came from `mlkit.metrics`. A metric
   computed inside a model file is an automatic `PENDING REVISION`.
6. **Threshold check.** Compare against the acceptance value in
   [SOURCE_LEDGER.md](SOURCE_LEDGER.md) — the one written **before** the code.
   A threshold edited after seeing results is a bug, and a serious one.
7. **Gap sweep.** List every assumption with no source anchor; each becomes an
   `UNVERIFIED` tag or a new gap ID.
8. **Tag and sign.** Write the row in §3 with the date and the run id.

**Hand-back rule.** The Auditor does not fix what they find. They log it and hand
it back. Mixing the roles destroys the independence that makes the tag worth
anything.

---

### 3. Status board

Update on every gate. `—` means not yet attempted.

#### Competencies

| ID | Competency | Status | Evidence (run id) | Date |
|---|---|---|---|---|
| C-01 | Linear algebra from scratch | — | | |
| C-02 | Gradient-based optimisation from scratch | — | | |
| C-03 | Probability, information, MLE | — | | |
| C-04 | Leak-proof classical pipeline | — | | |
| C-05 | Honest evaluation with CIs | — | | |
| C-06 | PyTorch without framework magic | — | | |
| C-07 | Paper implemented from the original text | — | | |
| C-08 | Sequence modelling with correct temporal validation | — | | |
| C-09 | Uncertainty with coverage guarantees | — | | |
| C-10 | Deliberate tuning and scaling | — | | |
| C-11 | Tracking, reproduction, handover | — | | |
| C-12 | Serving and latency measurement | — | | |
| C-13 | Explainability and governance | — | | |
| C-14 | Professional-grade code | — | | |

#### Competencies the source stack cannot certify

| ID | Competency | Status | Blocking gap |
|---|---|---|---|
| X-01 | Reinforcement learning theory | **UNVERIFIED** | GAP-04 — no RL source in the stack |
| X-02 | Distributed / multi-GPU training | **UNVERIFIED** | GAP-05 — no DDP/FSDP coverage |
| X-03 | Generalisation bounds (PAC, VC) | **UNVERIFIED** | GAP-06 — [MML] stops at MLE |
| X-04 | Container / CI-CD-grade production | **UNVERIFIED** | GAP-07 — [PTG p.385–407] is demo-grade |

These four stay `UNVERIFIED` permanently unless an external source is added to the
ledger. Claiming them without that source is the kind of quiet overstatement this
protocol is built to prevent.

#### Stage gates

| Gate | Requirement | Status | Date |
|---|---|---|---|
| Stage 0 | EXP-01 verified; lint, type, metric and leak tests green | **VERIFIED** — contract tests pass with the 11 Stage 1 gates correctly skipped, `ruff check`, `ruff format --check` and `mypy mlkit` are clean at 80 columns, and all four EXP-01 arms are VERIFIED after BUG-001 was repaired. | 2026-10-17 |
| Stage 1 | Math modules green; LR divergence study; GD arm matches normal equations to 1e-6 | — | |
| Stage 2 | EXP-02 leak-proof benchmark, nested CV, calibrated, reproduced | — | |
| Stage 3 | Hand-written training loop + one paper reproduced against its own number | — | |
| Stage 4 | EXP-04 rolling-origin benchmark with verified conformal coverage | — | |
| Stage 5 | Tracked, served, explained, governed model with a reproduced audit pack | — | |
| Capstone | Independent reader reproduces every headline number within its CI | — | |

#### Experiments

| Experiment | Arms | Status | Notes |
|---|---|---|---|
| EXP-01 baseline | normal_equations | **VERIFIED** | Seed 20260101, conda env `qwen35`: recovers the generator's weights to 0.0121 (< 0.1), and test RMSE 0.2564 sits at the irreducible error floor (noise sd 0.25). |
| EXP-01 baseline | gradient_descent | **VERIFIED** | Matches the closed-form solution to 6.22e-15 (gate: < 1e-6). Calculus and linear algebra agree. |
| EXP-01 baseline | sklearn_linear | **VERIFIED** | scikit-learn 1.9.1; matches the closed-form solution to 2.04e-14 (gate: < 1e-6). |
| EXP-01 baseline | torch_sgd | **VERIFIED** | torch 2.13.0 (CUDA 13.4) after BUG-001 was repaired; float32 mini-batch SGD matches the closed form to 0.0198 (gate: < 0.05, loosened for float32 arithmetic). |
| EXP-01 baseline | **overall** | **VERIFIED** | All five gates pass across all four methodologies. Run id `20261017T020040-exp01_baseline`. |
| EXP-01 baseline | **replay** | **VERIFIED** | Protocol §2.3 replay, 2026-10-17: re-run from the same config and seed in a separate session produced identical gate values (0.0121, 0.2564, 6.22e-15, 2.04e-14, 0.0198). Run id `20261017T022631-exp01_baseline`. Contract tests pass with the 11 Stage 1 gates skipping as designed; `ruff check` and `ruff format --check` clean over 23 files, `mypy` clean over its 10 source files (`mlkit` and `panel`). Re-checked independently 2026-10-17 02:30: the run directory is on disk and its five gate values match this row exactly. |
| EXP-02 tabular | — | — | |
| EXP-03 deep | — | — | |
| EXP-04 forecast | — | — | |
| Capstone | — | — | |

---

### 4. Bug log

One row per defect. Nothing is deleted; bugs are closed, not removed.

| ID | Date | Found in | Severity | Description | Anchor | Status | Fixed by |
|---|---|---|---|---|---|---|---|
| *(example)* BUG-000 | 2026-01-01 | `exp01` | high | Scaler fitted before the split, inflating RMSE by 11%. | [TSP p.159] | closed | run `2026…` |
| **BUG-001** | 2026-10-17 | conda env `qwen35`, surfaced by `exp01` torch arm | high | **Two torch distributions were installed on top of each other.** `site-packages` carried both `torch-2.10.0.dist-info` (conda-forge `pytorch 2.10.0 cuda130_mkl`, built from `C:/bld/libtorch_…`) and `torch-2.13.0+cu130.dist-info` (a pip CUDA 13.0 wheel). The package directory was a mixture: `torch/version.py` reported 2.10.0 while the compiled `_C` extension came from the other build. Activated, the import died at `torch._C.TensorBase` having no `align_as`; unactivated, earlier still, loading `torch\lib\shm.dll` (WinError 127). | environment, not a source rule | **closed** | Repaired 2026-10-17: removed both pip distributions (`torch`, `torchvision`, `torchaudio`) with no leftover files, then reinstalled from conda-forge as `pytorch 2.13.0 cuda134_mkl_py311` + `libtorch 2.13.0` + `torchvision 0.28.0 cuda130_py311`. Verified: `torch.cuda.is_available()` is True on an RTX 5070 Laptop GPU, a 1024×1024 CUDA matmul runs, and `transformers 5.17.0`, `accelerate 1.14.0`, `bitsandbytes 0.50.2`, `optimum`, `safetensors` and `sentencepiece` all still import. EXP-01's torch arm now VERIFIED. **Follow-on: GAP-15 (torchaudio).** |
| **BUG-002** | 2026-10-17 | `mlkit/report.py`, `mlkit/verify.py` | high | `env_info()` caught only `ImportError`, so a package that was installed but broken crashed the very report meant to describe the environment, and `mlkit.verify --env` died mid-boot. Now recorded as `BROKEN (<ErrorType>)`. | local | closed | this session |
| **BUG-003** | 2026-10-17 | `experiments/exp01_baseline.py` | high | An arm whose dependency was installed but broken raised through `run_experiment` and killed the whole run, so one bad package hid three healthy arms. Import failure now tags `UNVERIFIED` (absent) or `BUG` (present but broken), and a `BUG` arm drops the run to `PENDING REVISION`. | VERIFICATION.md §1 | closed | this session |
| **BUG-004** | 2026-10-17 | `pyproject.toml` | medium | The lint config contradicted its own cited source: ruff's default import sorting merges names onto one line, while [GPSG p.40 §3.13] requires one import per line except for `typing` and `collections.abc`. Config now encodes the rule it claims to enforce. | [GPSG p.40 §3.13] | closed | this session |

**Severity.** *critical* — a reported number is wrong; *high* — a gate can pass
while wrong; *medium* — correct but misleading; *low* — cosmetic or style.

**Rules.**
- A bug that makes a reported number wrong flips that number to `PENDING
  REVISION` immediately, and the number stays visible with its tag.
- A bug found after a gate passed **reopens the gate**.
- Every bug names the anchor it violates, so the fix can be checked against the
  source rather than against an opinion.

---

### 5. Known blind spots carried from the ledger

Repeated here so they stay in front of the Auditor at every review.

| ID | Blind spot | Review action |
|---|---|---|
| GAP-01 | [SKL] has no stable pagination | Accept heading anchors; reject page citations for it |
| GAP-02 | [TSP] anchors are PDF page indices | Do not cross-check against a print copy |
| GAP-03 | [TSP] repository URL reconstructed across a line break | Verify before cloning |
| GAP-04 | No RL theory in the stack | Any RL claim is `UNVERIFIED` |
| GAP-05 | No distributed training | Any multi-GPU claim is `UNVERIFIED` |
| GAP-06 | No generalisation bounds | Reject any bound-flavoured claim |
| GAP-07 | Production coverage is demo-grade | Deployment stays `UNVERIFIED` |
| GAP-08 | Leakage definitions differ across sources | [TSP p.158–164] arbitrates |
| GAP-09 | Metric definitions overlap and disagree | Reject any metric not from `mlkit.metrics` |
| GAP-10 | [GPSG] has no ML hygiene rules | Local conventions tagged as locally authored |
| GAP-11 | [30u30] code is un-audited | Outputs stay `PENDING REVISION` until matched to the paper |
| GAP-12 | No runtime baselines exist in any source | Record your own; never quote a source's timing |
| GAP-13 | **Closed** 2026-10-17 — `qwen35` now has the full stack: scipy, pandas, scikit-learn, matplotlib, pyarrow, statsmodels, pytest, ruff, mypy, and torch 2.13.0 with working CUDA | A broken package is tagged `BROKEN` by `verify --env` and `BUG` by the arm — never a silent skip |
| GAP-14 | `qwen35` is an existing environment, so the pinned versions in `requirements.txt` are **not** enforced there (numpy 2.4.6, scipy 1.17.1, pandas 3.0.6, sklearn 1.9.1 are all newer than the pins) | Results from it reproduce on this machine, not from a clean clone. For a reproducible run use the default `.venv` mode |
| **GAP-15** | **torchaudio is unavailable in a consistent CUDA build on this stack.** conda-forge ships torch 2.13 for py311 only as `cuda134`, while every conda-forge torchaudio CUDA build is `cuda130`, and torchaudio refuses a CUDA-version mismatch at import. The `gpu_cuda130` torch builds that would match live on anaconda `main`, which tops out at 2.9.1 for py311 and is excluded here by strict channel priority. | Audio models cannot be used in `qwen35` as it stands. Nothing in this program's stages 1–5 needs torchaudio, so no gate is blocked. Closing it means either moving the whole trio to pip's cu130 wheels or dropping to Python 3.10/3.12 where the builds line up |

---

*Protocol version 1.0. The Auditor owns this file.*

---


# BOOK FOUR — Testbed Operations

*How to boot and drive the runnable half: the one-button setup, the daily commands, the two rules, and the dashboard.*

---

The runnable half of the learning program. One command brings it up from
scratch; after that every experiment shares one loader, one scorer and one
set of splitters, so comparing two methodologies compares the methodologies
and nothing else.

### Boot

The script resolves an interpreter, installs what is needed, prints what this
machine can verify, runs the contract tests, and executes the baseline
experiment. Requires CPython 3.11 or newer: configs are TOML, read with the
standard library's `tomllib`.

**Clean and pinned** — creates `.venv` and installs the exact versions in
`requirements.txt`. Use this when a result has to reproduce elsewhere.

```powershell
.\setup.ps1
```

```bash
./setup.sh
```

**An environment you already have** — installs only the packages that are
absent and leaves every installed version alone. Pins are *not* enforced, so
results reproduce on that machine rather than from a clean clone; say so in
the run report (ledger GAP-14).

```powershell
.\setup.ps1 -CondaEnv qwen35
.\setup.ps1 -Python C:\Python312\python.exe
```

```bash
./setup.sh --conda-env qwen35
./setup.sh --python /usr/bin/python3.12
```

Pass `-Minimal` / `--minimal` for the core packages only; the deep-learning
arms then report `UNVERIFIED`, which is the designed behaviour rather than a
failure.

#### Missing, broken, and the difference between them

A package that is **absent** makes its arm `UNVERIFIED` — an honest skip. A
package that is **installed but fails to import** makes its arm `BUG`, keeps
the error text in the report, and drops the whole run to `PENDING REVISION`.
The two are not the same condition and the testbed never conflates them:

```bash
python -m mlkit.verify --env       # inventory, with BROKEN called out
python -m mlkit.verify --missing   # pip names of absent packages only
```

### Layout

```
testbed/
  mlkit/
    data.py        the shared input field -- the only way data enters
    metrics.py     the single measurement tools -- the only scorer
    splits.py      splitters plus the assertions that catch leaks
    report.py      run reports: the audit pack
    config.py      the control panel (--set, --sweep, --compare)
    verify.py      the Auditor's entry point
  experiments/
    exp01_baseline.py   the single baseline experiment
  panel/
    reports.py     parses runs/ into one JSON-safe payload
    serve.py       the read-only dashboard, `python -m panel.serve`
    static/        vanilla HTML/CSS/JS, no build step
  exercises/
    stage1_linalg.py    Stage 1: you implement these
  tests/
    test_metrics.py     the measurement contract
    test_splits.py      the split contract
    test_leakage.py     the leakage contract, demonstrated on noise
    test_stage1_gates.py  Stage 1 acceptance gates (skip until implemented)
    test_panel_reports.py the panel's parsing contract
  configs/exp01.toml    controls for the baseline
  runs/                 one audit pack per run (created on first run)
```

### Daily commands

```bash
# run the baseline and write an audit pack
python -m experiments.exp01_baseline --config configs/exp01.toml

# change a parameter without touching the source
python -m experiments.exp01_baseline --config configs/exp01.toml \
    --set arms.gradient_descent.lr=0.3

# scan a parameter range; divergence above the stability limit is the
# point of this one [MML p.381]
python -m experiments.exp01_baseline --config configs/exp01.toml \
    --sweep arms.gradient_descent.lr=0.01,0.2,0.5,0.9,1.1

# read an arm's implementation and its page anchors
python -m experiments.exp01_baseline --config configs/exp01.toml \
    --show-source gradient_descent

# read the runs in a browser (read-only dashboard)
python -m panel.serve

# what can this machine verify?
python -m mlkit.verify --env

# re-check a finished run as the Auditor, from the run directory alone
python -m mlkit.verify runs/<run-id>

# contracts, and the Stage 1 gates
pytest tests -q
pytest tests/test_stage1_gates.py -v

# the code contract [GPSG]
ruff check . && ruff format --check . && mypy mlkit
```

### The two rules

1. **Nothing computes its own metric.** Every number comes from
   `mlkit.metrics`, because three of the source books define overlapping
   metrics that disagree at the edges (ledger GAP-09).
2. **Nothing loads its own data.** Every array comes from
   `mlkit.data.load_dataset`, so two methodologies are always compared on
   byte-identical inputs.

Break either one and the comparison tables stop meaning anything, quietly.

### What a run leaves behind

```
runs/20261017T005554-exp01_baseline/
  config.json        the resolved config, including any --set overrides
  report.json        machine-readable: arms, gates, gaps, environment,
                     and created_at, the time the run was written
  report.md          human-readable summary with status tags
  env.txt            interpreter, platform, package versions
  dataset_card.json  name, shape, seed, checksum, provenance
```

That directory is the audit pack [SKL ch.14]. If a reviewer cannot tell what
you ran from those files alone, the run is not finished.

### Panel

A read-only dashboard over `runs/`. Standard library only: a small static
server plus vanilla HTML, CSS and JS. No build step, no database, no new
runtime dependency.

```bash
# open the dashboard against the existing runs/
python -m panel.serve

# another port, no browser window, no request log
python -m panel.serve --port 9000 --no-browser --quiet

# the same answers in a terminal; exits 1 if any run is not VERIFIED
python -m panel.serve --check
```

| View | What it answers |
|---|---|
| Status board | what was run, when, on which data, and is it trusted |
| Run detail | arms with 95% CIs and wall-clock seconds, gates with their page anchors, split boundaries, arm notes, gaps, environment, resolved config |
| Compare | two or more runs of one experiment: config diff beside the primary metric, so a parameter change is traceable to a metric change |
| Sweeps | runs differing in exactly one config key, that key plotted against the primary metric |
| Tags | what the four tags mean, and the two markers that are not tags |

Five rules hold it together.

1. **No number without its tag.** Every metric reaches the page through
   one function that cannot render a value without the status of the arm
   that produced it. There is deliberately no helper that returns a bare
   formatted number, because the moment one exists some table uses it.
2. **Nothing is invented.** A field the report does not carry is reported
   missing, never defaulted. A `report.json` that will not parse is
   marked `UNREADABLE`, which is a parse outcome and emphatically not a
   fifth status tag: the panel does not get to decide a run's status. An
   arm the config declared but the report never produced is shown as a
   dropped arm rather than omitted.
3. **Unverified stays in view.** A banner above every view names every
   run that is not `VERIFIED`, including unreadable ones. There is no
   filter that takes it away.
4. **Divergence is a result, not an axis.** A metric that is non-finite,
   or above `1e6` in magnitude, is drawn in a separate band above the
   plot with no position on the axis at all, and the series line breaks
   there. Otherwise one `rmse = 4.9e+65` flattens every other point into
   a single line. Divergence above the stability limit is a required
   result of the `lr` sweep, not a failure [MML p.381], [MML p.389].
5. **It is a viewer.** The server answers `GET` and `HEAD` and nothing
   else, and binds to loopback. Nothing in the panel edits a run, a
   threshold or a status -- editing a threshold after seeing a result is
   the bug this program exists to prevent (VERIFICATION.md §2.6).

The browser is never handed a raw `report.json`. A diverged run writes
`NaN`, which Python's `json` module emits happily and the browser's
`JSON.parse` rejects outright. `panel/reports.py` normalises every pack
into one JSON-safe payload served at `/api/runs.json`, rebuilt from disk
on each request, so a run that finished a second ago appears on the next
refresh.

#### Feeding the sweep view

`--sweep` prints a parameter/metric curve but writes no audit packs, so
the sweep view reads runs made with repeated `--set`:

```bash
for lr in 0.01 0.05 0.2 0.5 0.9 1.1; do
  python -m experiments.exp01_baseline --config configs/exp01.toml \
      --set arms.gradient_descent.lr=$lr
done
```

The panel groups those runs by the one key that varies. Runs differing
in two keys form no sweep, because such a pair explains nothing.

#### One field added to the audit pack

`report.json` gained `created_at` (ISO 8601), written by
`mlkit.report.write_run` from the same moment it stamps the run
directory with. The change is additive: reports written before it exists
parse exactly as before, and the panel falls back to the directory name,
marking those timestamps with `*` on the status board. The reason is the
audit-pack rule itself -- a reviewer holding `report.json` alone should
not have to reconstruct when the run happened from a directory name,
which is a naming convention rather than data.

---
