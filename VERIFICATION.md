# Verification Protocol, Status Board, and Bug Log

> **Step 3 of the Prompt Guidelines:** *"Set up an independent review stage: let
> reviewers record bugs, and leave all unverified results clearly marked with
> warning tags."*

This file is maintained by the **Auditor** role only. The Coder may add bug
entries; the Coder may not change a status tag.

---

## 1. Status tags

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

## 2. The review stage

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

## 3. Status board

Update on every gate. `—` means not yet attempted.

### Competencies

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

### Competencies the source stack cannot certify

| ID | Competency | Status | Blocking gap |
|---|---|---|---|
| X-01 | Reinforcement learning theory | **UNVERIFIED** | GAP-04 — no RL source in the stack |
| X-02 | Distributed / multi-GPU training | **UNVERIFIED** | GAP-05 — no DDP/FSDP coverage |
| X-03 | Generalisation bounds (PAC, VC) | **UNVERIFIED** | GAP-06 — [MML] stops at MLE |
| X-04 | Container / CI-CD-grade production | **UNVERIFIED** | GAP-07 — [PTG p.385–407] is demo-grade |

These four stay `UNVERIFIED` permanently unless an external source is added to the
ledger. Claiming them without that source is the kind of quiet overstatement this
protocol is built to prevent.

### Stage gates

| Gate | Requirement | Status | Date |
|---|---|---|---|
| Stage 0 | EXP-01 verified; lint, type, metric and leak tests green | **VERIFIED** — contract tests pass with the 11 Stage 1 gates correctly skipped, `ruff check`, `ruff format --check` and `mypy mlkit` are clean at 80 columns, and all four EXP-01 arms are VERIFIED after BUG-001 was repaired. | 2026-10-17 |
| Stage 1 | Math modules green; LR divergence study; GD arm matches normal equations to 1e-6 | — | |
| Stage 2 | EXP-02 leak-proof benchmark, nested CV, calibrated, reproduced | — | |
| Stage 3 | Hand-written training loop + one paper reproduced against its own number | — | |
| Stage 4 | EXP-04 rolling-origin benchmark with verified conformal coverage | — | |
| Stage 5 | Tracked, served, explained, governed model with a reproduced audit pack | — | |
| Capstone | Independent reader reproduces every headline number within its CI | — | |

### Experiments

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

## 4. Bug log

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

## 5. Known blind spots carried from the ledger

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
