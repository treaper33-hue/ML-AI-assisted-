# ML Mastery — a verifiable path to professional capability

A learning program built from five sources and one rule: **nothing counts as
learned until a numeric gate passes and the result is tagged.**

## Start here

```powershell
cd testbed; .\setup.ps1 -CondaEnv qwen35
```

(Plain `.\setup.ps1` builds an isolated, pinned `.venv` instead — use that when
a result has to reproduce on someone else's machine.)

Then read [ML_LEARNING_PROGRAM.md](ML_LEARNING_PROGRAM.md) from Part 0.

## The four documents

| File | What it is | Who owns it |
|---|---|---|
| **[ML_LEARNING_PROGRAM.md](ML_LEARNING_PROGRAM.md)** | The curriculum: 6 stages, 30+ modules, each with build / train / execute / gate / blind spots | you, as Analyst and Coder |
| **[SOURCE_LEDGER.md](SOURCE_LEDGER.md)** | Every algorithm extracted from the books with its **page number**, its test parameters, and 13 named gaps | you, as Analyst |
| **[VERIFICATION.md](VERIFICATION.md)** | The review protocol, status board and bug log | you, as Auditor |
| **[testbed/](testbed/)** | The runnable environment: one loader, one scorer, one set of splitters, one baseline experiment | you, as Interface |

## Where it comes from

| Source | Role in the program |
|---|---|
| *Mathematics of Machine Learning* (Danka) | Stage 1 — linear algebra p.3–262, calculus p.267–428, multivariable p.435–484, probability p.491–629 |
| *Prompting Scikit-Learn for ML* | Stage 2 — pipelines, leak-safe splits, evaluation, tuning, explainability, governance |
| *PyTorch: The Practical Guide* (Gollnick) | Stages 3 and 5 — tensors p.42, training p.45–80, vision p.127, Lightning p.351, tracking p.363, deployment p.385 |
| *Time Series with PyTorch* (Davidson & Ma) | Stage 4 — diagnostics, temporal evaluation p.112–166, conformal prediction p.309–348, sequence models |
| *Google Python Style Guide* | The code contract — 80 columns p.20, docstrings p.27, naming p.43, types p.46 |
| `papers/01..30` (this repo) | Paper implementations woven into Stages 3–5, each held at `PENDING REVISION` until reproduced against the paper's own number |

## What makes this different from a reading list

1. **Every claim has a page number.** If the ledger has no anchor for it, it
   does not go in the program.
2. **Every result has a tag** — `VERIFIED`, `PENDING REVISION`, `UNVERIFIED`
   or `BUG` — and an untagged number does not exist.
3. **Four gaps are admitted up front.** Reinforcement learning, distributed
   training, generalisation bounds and container-grade production are **not**
   covered by these sources, so those competencies stay `UNVERIFIED` rather
   than being quietly claimed.
4. **One scorer, one loader.** Three of the books define overlapping metrics
   that disagree at the edges; the disagreements are resolved once, in code,
   with the edge cases as tests.
5. **It boots with one command** on a clean machine, which is the only way
   anyone — including you in six months — can check your work.

## Status right now

Last boot: 2026-10-17, conda env `qwen35` (Python 3.11.16).

| Item | State |
|---|---|
| EXP-01 — all four arms | **VERIFIED** — closed form, hand-written gradient descent, scikit-learn and PyTorch all agree with the analytic optimum (6.2e-15, 2.0e-14 and 0.0198 against a float32 gate of 0.05), with test RMSE at the irreducible error floor |
| torch in `qwen35` | **repaired** (BUG-001) — two stacked distributions removed and reinstalled as torch 2.13.0 with CUDA 13.4; `torch.cuda.is_available()` is True on an RTX 5070, and transformers, accelerate and bitsandbytes still import |
| torchaudio | **GAP-15** — no consistent CUDA build exists for Python 3.11 on conda-forge; nothing in stages 1–5 needs it, so no gate is blocked |
| Contract tests (metrics, splits, leakage) | **passing**; the 11 Stage 1 gates skip, as designed |
| Code contract | `ruff check`, `ruff format --check` and `mypy mlkit` all clean at 80 columns |
| Stage 1 exercises | stubs with gates — the gates skip until you implement them, and a skip is not a pass |
| Stages 2–5 experiments | specified in the program, not yet built — that is your work |

Reproducibility caveat (GAP-14): `qwen35` keeps its own newer package versions,
so the pins in `requirements.txt` are not enforced there. Results from it
reproduce on this machine; for a clean-clone guarantee, boot the `.venv` mode.

---

## This repository

Live at **<https://github.com/treaper33-hue/ml-mastery>** (public).

It deliberately does **not** include the surrounding `30u30` clone it
was written beside, whose content is licensed CC BY-NC-ND — this work
references those papers, it does not redistribute them.

### Open any notebook in Colab

Every notebook carries an **Open in Colab** badge at the top, and the
URLs work with no sign-in:

```
https://colab.research.google.com/github/treaper33-hue/ml-mastery/blob/main/notebooks/part1_mathematical_engine.ipynb
```

Swap the filename for any of the nine — `part1`–`part4` for the
exercises, `project1`–`project5` for the projects.

If the repository is ever renamed or forked, one command rewrites all
nine badges and both READMEs:

```bash
python notebooks/set_colab_links.py <owner>/<repo>
```

### What is not committed

`.gitignore` excludes `runs/` wherever it appears — audit packs are
reproducible from their config, and their `env.txt` records absolute
paths from the machine that produced them. Also excluded: virtual
environments, caches, `*.pt` checkpoints, and `book/build/`. The PDF
itself *is* committed, because it is the deliverable.

### Licence

No licence file is included, which under GitHub's terms means all
rights are reserved: others may view and fork within GitHub, but have
no licence to reuse the work. Add one if you want different terms.
