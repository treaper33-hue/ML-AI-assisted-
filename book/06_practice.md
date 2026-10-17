# Part V — Professional Practice

A correct model that nobody can rerun, read, or trust is not a professional
deliverable. This part is about the difference between code that works once
and work that a team can own.

---

# 26 · The code contract

Every listing in this book follows the Google Python Style Guide. The rules
that matter most in machine-learning code, with their anchors:

| Rule | Anchor | Why it matters here |
|---|---|---|
| 80-column lines | *[GPSG p.20 §3.2]* | side-by-side diffs and review |
| 4 spaces, never tabs | *[GPSG p.23 §3.4]* | — |
| Docstrings on every public module, class and function | *[GPSG p.27 §3.8]* | — |
| Args / Returns / Raises in function docstrings | *[GPSG p.29 §3.8.3]* | the shape of your data belongs here |
| Type annotations | *[GPSG p.19 §2.21, p.46 §3.19]* | catches `ndarray` vs `Tensor` before runtime |
| `module_name`, `ClassName`, `function_name`, `GLOBAL_CONSTANT` | *[GPSG p.43 §3.16]* | — |
| `main()` plus `if __name__ == '__main__':` | *[GPSG p.45 §3.17]* | makes every script importable and testable |
| Rethink functions past ~40 lines | *[GPSG p.46 §3.18]* | training loops bloat silently |
| No bare `except:` | *[GPSG p.5 §2.4]* | a swallowed exception becomes a silent NaN |
| `TODO(username):` | *[GPSG p.39 §3.12]* | — |

One documented exemption: inside a derivation, short mathematical names
(`A`, `x`, `lr`, `mu`) matching the source notation are allowed and preferred
*[GPSG §3.16.5]*. Local only — never a public API name.

```python
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes the root mean squared error.

    Args:
        y_true: Ground-truth values, shape (n,).
        y_pred: Predicted values, shape (n,).

    Returns:
        The RMSE as a float.

    Raises:
        ValueError: If the shapes do not match.
    """
```

That docstring is not decoration. It states the shapes — the single most
common source of silent bugs in array code — and it names the failure mode.

## 26.1 Conventions the style guide does not cover

The Google guide is about style, not about machine learning. These seven rules
are local to this text and carry no external authority; they are stated so you
can argue with them rather than absorb them unawares.

1. **One seed per run**, in the config, recorded in the report.
2. **Notebooks explore; modules decide.** A number that reaches a report comes
   from a module invoked by a script, never from notebook state.
3. **Configs are data.** No experiment reads a value not in its config.
4. **Datasets are addressed by name and checksum**, never by path.
5. **Every run writes an audit pack** — config, report, environment, dataset
   card, plots.
6. **Failures are results.** A model losing to the baseline is reported with
   the same prominence as a win.
7. **A number without a tag does not exist.**

---

# 27 · The method

## 27.1 Four roles, never two at once

The accompanying testbed is organised around four roles. You will play all
four; the discipline is never to play two simultaneously on the same artifact.

| Role | Mandate | Forbidden |
|---|---|---|
| **Analyst** | Extract the mathematics, assumptions and acceptance thresholds from the source. | Writing implementation code. |
| **Coder** | Turn that into working, lint-clean, typed code. | Choosing the metric, or editing the threshold. |
| **Auditor** | Re-derive the result independently; try to break it. | Fixing what they find — hand it back. |
| **Interface** | Build the controls, tables and plots. | Changing numbers to improve a plot. |

Sleep between Coder and Auditor on the same artifact. Audit your own code ten
minutes after writing it and you reread your intentions, not the code.

## 27.2 Thresholds before code

The acceptance threshold is written **before** the implementation, by the
Analyst, from the source. A threshold adjusted after seeing the result is not
a threshold; it is a rationalisation, and it is the most serious defect this
method is built to prevent.

## 27.3 Status tags

| Tag | Meaning | Citable? |
|---|---|---|
| **VERIFIED** | Reproduced from a clean run; the numeric gate passed. | yes |
| **PENDING REVISION** | A gate failed, or an open bug touches it. | no |
| **UNVERIFIED** | Ran, but has no numeric gate — a visual result, a missing dependency, a competency outside the sources. | only with the tag attached |
| **BUG** | A defect with a log entry; blocks the gate. | no |

Any `UNVERIFIED` or `PENDING` number must carry its tag *everywhere it
appears* — report, plot title, table, conversation. A number that travels
without its tag is the failure mode the whole structure exists to prevent.

## 27.4 The audit pack

Every run writes a directory containing the resolved config, the report in
both machine and human form, the environment (interpreter, platform, package
versions), the dataset card (name, shape, seed, checksum, provenance), and the
plots. The test: **a stranger, given only that directory, can rerun the work
and match your numbers.** If they cannot, the run is not finished, however
good the model is.

---

# 28 · From model to system

## 28.1 Tracking

Log parameters, metrics, artifacts and the environment for every run
*[PTG p.363 TensorBoard, p.372 MLflow, p.377 Weights & Biases]*. The gate is
blunt: delete a run's local files and rebuild the result from the tracking
record alone.

## 28.2 Serving

Start local *[PTG p.387]*. The things that actually bite in production are
rarely the model:

- **Input validation** — reject malformed requests loudly, at the boundary.
- **Training/serving skew** — the same preprocessing code must run in both
  places. This is the single most common production failure, and shipping the
  fitted `Pipeline` object rather than reimplementing the transforms is the
  defence.
- **Latency** — measure p50 *and* p95; the average hides the tail users feel.
- **Rollback** — have one before you need one.
- **Monitoring** — input drift, output drift, and the realised metric once
  labels arrive.

## 28.3 Explainability

| Tool | Answers | Caveat |
|---|---|---|
| Permutation importance | which features the model relies on | compute on **held-out** data, and note it splits credit arbitrarily between correlated features |
| Partial dependence | the average direction and size of an effect | assumes feature independence |
| ICE curves | per-sample effects | plot *with* PDP — the average hides heterogeneity |
| Standardised coefficients | effect size in a linear model | only meaningful after scaling |
| Local attributions | why *this* prediction | unstable; never present a single run as definitive |

Model-agnostic explanations are approximations of the model, not of reality.
They describe what the model does, which is not the same as what is true.

## 28.4 Governance

A model card: intended use, training data and date, metrics with intervals and
on which slices, known failure modes, and who to contact. A fairness
assessment naming the protected attribute and the harm considered. An incident
plan with a rollback trigger. These are not bureaucracy — they are what makes
the difference between a model you deployed and a model you are accountable
for.

---

# 29 · The route through

## 29.1 Twenty-four weeks

| Weeks | Material | You can prove |
|---|---|---|
| 0 | Testbed, code contract, metrics and leakage contracts (Ch 26–27) | the environment boots and the gates run |
| 1–4 | Part I (Ch 1–8) | the mathematics, in code you wrote |
| 5–8 | Part II (Ch 9–15) | a leak-proof pipeline with honest intervals |
| 9–13 | Part III (Ch 16–21) | a network trained from a hand-written loop |
| 14–18 | Part IV (Ch 22–25) | a forecast with verified interval coverage |
| 19–22 | Part V (Ch 26–28) | a tracked, served, explained model |
| 23–24 | Capstone | someone else reproduces every number |

Work the matching project in Part VI at the end of each block: project 1 after
week 8, project 2 after week 13, projects 3 and 4 after week 18. They are the
rehearsal for the capstone, and each one takes under a minute to run.

Compressed to fourteen weeks at double the hours: trim Chapter 3, Chapter 15
and Chapter 19 first. **Never compress** the metric contract, the leakage
chapter, temporal evaluation or conformal coverage. Everything else can be
relearned later; those four are what make every other number true or false.

## 29.2 The daily shape

| Minutes | Role | Activity |
|---|---|---|
| 30 | Analyst | read the anchored pages; write the threshold down first |
| 60 | Coder | implement, test, lint |
| 20 | Auditor | run the gate, tag the result, log what broke |
| 10 | Interface | update the board, commit the run |

## 29.3 The capstone

One system a team could inherit: data through a single loader with a card and
checksum; diagnostics with disagreements recorded; at least four arms
including a naive baseline on identical rolling-origin folds; one scorer;
conformal intervals with measured coverage including under a deliberate shift;
the leakage suite green; permutation importance and PDP/ICE; a served endpoint
with latency numbers; an audit pack; and a written list of everything still
`UNVERIFIED`.

It is finished when a person who has not seen your work clones the repository,
runs one command, and reproduces every headline number within its stated
interval.

---
