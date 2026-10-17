# Part VI — Five Worked Projects

Everything so far has been pieces. This part is five complete systems,
each a single runnable file in `ml-mastery/projects/`, each going from raw
data to a tagged result with an audit pack. Run them before you read about
them:

```bash
cd ml-mastery/projects
python run_all.py
```

On the machine this text was written on: **5/5 verified, 39 gates, 90
seconds**, no downloads.

| # | Project | Chapters | Runtime |
|---|---|---|---|
| 1 | Tabular classification, end to end | 9–14 | ~40 s |
| 2 | A convolutional classifier, trained by hand | 16–19 | ~10 s |
| 3 | Forecasting with real intervals | 22–25 | ~2 s |
| 4 | A VAE with the ELBO taken apart | 5, 8, 20 | ~20 s |
| 5 | Serving a model, and watching it afterwards | 28 | ~9 s |

> **Read this part differently.** The interesting content is not that the
> projects pass. It is the three places where they *did not*, where the
> gate turned out to be wrong rather than the code. Those are in §30.4,
> §31.3 and §33.2, and they are the most honest pages in this book.

---

# 30 · Project 1 — Tabular classification

## 30.1 The data is the lesson

The generator is not a convenience; it is where the difficulty lives. It
produces an 18% positive rate, values missing *not* at random (dormant
accounts have no session data), an 18-level region column with a long
tail, and — the important one — **several rows per customer**.

That last pathology is what makes the project real. Rows are not
independent, so a plain `KFold` would put the same customer in train and
test, and the model would be scored on its ability to recognise people it
has already seen. The defence is a group-aware splitter, and the project
asserts the result:

```python
splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
dev_idx, test_idx = next(splitter.split(frame, labels, groups))
overlap = set(groups[dev_idx]) & set(groups[test_idx])
# gate: len(overlap) == 0
```

## 30.2 The threshold is a business parameter

Chapter 9 derived p\* = C_FP/(C_FP + C_FN). With a £5 false alarm and a
£120 missed churn, that is 0.04 — not 0.5. The project searches for the
cost-minimising threshold on validation data and finds **0.033**, close to
theory and different from it because the model's probabilities are not
perfect.

The number that matters is what it saves:

| Threshold | Test cost |
|---|---|
| 0.500 (the default nobody chose) | £15,055 |
| 0.033 (chosen from the costs) | £2,460 |

**Six times the loss, from a single line nobody thought to question.** No
modelling change produces an improvement that size.

## 30.3 Nested CV, and reading the gap

The project runs group-aware nested CV across three model families. The
optimism gap — flat minus nested — is the selection bias of Chapter 14,
made visible. In `--quick` mode it is near zero, which is itself the
lesson: with four configurations there is almost nothing to overfit.
Widen the grid and watch it grow.

## 30.4 Where the gate was wrong

The first version compared the nested score against `search.best_score_`
from the *inner* loop, and the gate "nested ≤ flat" failed: the nested
score came out **higher**.

That is not selection bias running backwards. The inner CV trains on
three-quarters of an outer training fold, so its models see less data and
score worse for a reason that has nothing to do with optimism. The two
numbers were not comparable.

The fix was to measure the flat score on the **same folds**:

```python
# Select and report on the identical outer folds -- the mistake the
# nested procedure exists to correct.
flat = GridSearchCV(pipeline, grid, scoring='average_precision',
                    cv=outer_cv, n_jobs=-1, refit=False)
flat.fit(x, y, groups=groups)
```

**The code was right and the gate was wrong.** That distinction is worth
more than the project: a failing check is a claim about your
understanding, not only about your code.

---

# 31 · Project 2 — A convolutional classifier, trained by hand

## 31.1 Check the gradients before you trust the training

Before a single epoch, the project verifies autograd against central
differences on the real model, in double precision with dropout off:

```
worst relative gap autograd vs central difference: 7.31e-09
```

Chapter 16 explained why both details matter: dropout makes the function
stochastic so the comparison is meaningless, and float32's ε ≈ 10⁻⁷ puts
the round-off floor above the discrepancy you are hunting.

## 31.2 Measure the learning rate; do not guess it

The LR range test sweeps geometrically from 1e-4 to 1.0 and records the
loss. The minimum lands near 0.73; the project takes an order of
magnitude below it, 0.073, and trains. This is the 2/L edge of Chapter 6
appearing as an empirical curve.

Then three checks that most projects skip:

| Check | Result |
|---|---|
| Determinism: a second run at the same seed | identical losses |
| Checkpoint round-trip: reloaded model's logits | `0.00e+00` difference |
| Temperature scaling changes accuracy | `0.9694` → `0.9694`, exactly |

The third is a *theoretical* gate: dividing logits by a positive constant
cannot change their ranking, so accuracy must be unchanged to the last
bit. If it moves, the implementation is wrong — and the gate would catch
it.

## 31.3 Where the gate was wrong, again

The calibration gate said "test ECE after ≤ ECE before". It failed:
0.0140 → 0.0145.

The model was already well calibrated. Temperature was fitted on 288
validation points, and on 360 test points the ECE's own sampling noise is
about 1/√360 ≈ 0.05 — eighty times the movement being treated as a
regression. The original gate was testing noise.

The replacement states what is actually claimable:

| Gate | Observed |
|---|---|
| ECE improves on the data T was fitted on | 0.0193 → **0.0044** |
| Test ECE moves less than 1/√n = 0.053 | +0.0006 |

**A calibration method that does not help an already-calibrated model is
not a bug.** Reporting that honestly is the job.

---

# 32 · Project 3 — Forecasting with real intervals

## 32.1 Diagnostics first

ADF and KPSS are both run and **both outcomes recorded** — the discipline
of Chapter 22, enforced in code:

```
ADF p=0.9206  KPSS p=0.0100
reading: both agree: non-stationary
```

Four combinations are possible and the project names all of them; picking
whichever test suits your plan is the quiet dishonesty that nothing in the
output would otherwise reveal.

## 32.2 Baselines, and what beating them means

| Model | MASE | RMSSE |
|---|---|---|
| **ridge on lags** | **0.744** | 0.760 |
| gradient boosting on lags | 0.883 | 0.901 |
| seasonal naive | 1.100 | 1.087 |
| mean | 1.989 | 1.929 |
| naive | 2.771 | 2.687 |
| drift | 3.095 | 2.961 |

Ridge on lag features beats the best baseline by 32%. Note that the
simplest model wins — and note how badly drift does, which tells you the
series is not a random walk, consistent with the diagnostics.

Fold boundaries are printed and asserted, so a reviewer can verify the
evaluation without reading the code:

```
fold 0: train [0,499] < test [500,527]  (series t=528..555)
fold 1: train [28,527] < test [528,555] (series t=556..583)
```

## 32.3 The headline: coverage fails exactly where it must

Split conformal at α = 0.1, calibrated on held-out training data:

| Segment | n | Coverage |
|---|---|---|
| pre-shift | 410 | **0.946** |
| post-shift | 350 | **0.734** |

Target was 0.90. Before the regime shift, coverage holds. After it,
**twenty-one points of coverage are gone**.

Chapter 25 proved the guarantee from exchangeability alone. A permanent
level shift destroys exchangeability, so the guarantee evaporates — and it
evaporates precisely when you most needed a trustworthy interval, because
something changed.

This is the single most important result in the four projects. It is why
coverage must be *measured on a future window*, not assumed from the
theorem, and why the project reports interval width and Winkler score
beside coverage: an infinitely wide interval covers everything.

---

# 33 · Project 4 — A VAE with the ELBO taken apart

## 33.1 The beta sweep

| β | Reconstruction | KL | Active units | Reading |
|---|---|---|---|---|
| 0.0 | 22.66 | 65.36 | 8/8 | no KL penalty: a plain autoencoder |
| 0.5 | 24.45 | 1.91 | 1/8 | both terms alive |
| 1.0 | 26.25 | 0.45 | 0/8 | posterior collapse |
| 4.0 | 27.01 | 0.003 | 0/8 | posterior collapse |
| 20.0 | 27.02 | 0.0002 | 0/8 | posterior collapse |

Read the summed loss alone and these five models look broadly similar.
Read the terms apart and they are **completely different models**: one is
an unregularised autoencoder, one is balanced, three have switched their
latent off entirely.

That is why `elbo_terms` returns a tuple rather than a loss.

## 33.2 Where the gate was wrong, a third time

The first version used summed **squared error** for reconstruction, and
the β = 1 model collapsed: KL fell to 0.027, zero active units,
reconstruction worse than PCA.

The model was not broken. It was correctly minimising a badly scaled
objective. Squared error over 64 pixels in [0, 1] is of order 5; the KL
is measured in **nats** and wants tens. Against that exchange rate, the
cheapest move is to switch the latent off and accept the reconstruction
penalty.

The fix is the one Chapter 8 implies: use the **Bernoulli likelihood**,
which is both the correct likelihood for intensities in [0, 1] and
measured in nats, so the two terms are commensurable.

> **Posterior collapse is often a units bug, not a modelling mystery.**
> The project keeps `likelihood='gaussian'` available so you can
> reproduce the failure deliberately.

## 33.3 The fair baseline

PCA is the MSE-*optimal* linear map at a given rank (Chapter 5). Judging
a BCE-trained model by MSE compares two objectives and calls the
difference a result, so the comparison autoencoder is trained on squared
error too:

| Model | Test reconstruction (MSE) |
|---|---|
| PCA, 8 components | 1.564 |
| autoencoder, β = 0, trained on MSE | **1.430** |
| VAE, β = 1 | 2.836 (2/8 units active) |

Nonlinearity beats the linear optimum, as it should. The VAE is worse —
**and that is the correct outcome**, because it is paying reconstruction
for a latent you can sample from. The gate says `vae_pays_for_its_prior`
rather than pretending the VAE should win.

---

# 34 · Project 5 — Serving a model, and watching it afterwards

The project the other four lead to. A model that never leaves the
notebook has not been tested against the thing that actually breaks
models in production, which is rarely the model.

The service is real: a socket on loopback, a `ThreadingHTTPServer`, real
`urllib` requests. Nothing is mocked, so the latencies are measured
rather than asserted.

## 34.1 Ship the pipeline, not the recipe

The headline result is a zero:

```
max |served - in-process| over 50 rows: 0.00e+00
```

**Training/serving skew** is the most common production failure in
applied machine learning: the preprocessing is reimplemented in the
serving layer — in another language, by another team, six months later —
and the two implementations drift apart silently. Predictions in
production stop matching the ones that were validated, and no test
catches it because each half is correct on its own.

The defence is structural rather than vigilant. Chapter 11 put every
parameterised transform inside one `Pipeline` object; this project ships
*that object*, so there is no second implementation to disagree with the
first. The gate demands exact equality, not approximate:

```python
Gate('no_training_serving_skew',
     'served predictions equal in-process predictions',
     f'{skew:.2e}', skew < 1e-12)
```

## 34.2 Reject at the boundary

| Request | Response |
|---|---|
| missing `tenure_months` | 400 `missing fields: tenure_months` |
| `monthly_charge: "free"` | 400 `monthly_charge must be numeric, got str` |
| `days_since_login: Infinity` | 400 `days_since_login must be finite` |
| a JSON array instead of an object | 400 `body must be a JSON object` |
| a valid row | 200 `p=0.0316 (v2-boosting)` |

Four ways to be wrong, four 400s, no 500s and — crucially — no silently
imputed prediction. A service that quietly guesses when the input is
malformed produces numbers nobody can audit later.

## 34.3 Latency, and where it actually goes

```
server-side p50 24.13 ms   p95 26.42 ms   p99 28.57 ms
```

Report **p95**, not the mean: the mean is dragged around by the body of
the distribution while users experience the tail.

The more useful observation is that **almost none of those 24 ms is the
model**. A single-row `predict_proba` through a `ColumnTransformer`
spends its time constructing a one-row DataFrame, dispatching through
pandas, and running transformers designed for batches. Batching requests
moves this by an order of magnitude; optimising the model would move it
by nothing.

That is the general lesson of production performance work: **measure
before you optimise, or you will optimise the wrong layer** with great
diligence.

## 34.4 Watching the inputs, because the labels are late

Labels arrive weeks after predictions, so the realised metric is the
*last* signal you get, not the first. Input monitoring is what you have
in the meantime.

The project uses the **population stability index**, which bins the live
sample by the training data's quantiles and sums

<div class="eq">PSI = Σ<sub>bins</sub> (p<sub>live</sub> − p<sub>ref</sub>) · ln(p<sub>live</sub> ⁄ p<sub>ref</sub>)</div>

The conventional readings: below 0.1 nothing material, 0.1–0.25
moderate, above 0.25 significant. Note the form — it is a symmetrised
cousin of the KL divergence of Chapter 8, and non-negative for the same
reason.

| Feature | In-distribution | After a deliberate shift |
|---|---|---|
| support_tickets | 0.0037 | **7.2187** |
| days_since_login | 0.0177 | **6.6652** |
| tenure_months | 0.0207 | **1.1913** |
| avg_session_minutes | 0.0169 | 0.0169 |
| monthly_charge | 0.0111 | 0.0111 |

Zero false alarms on in-distribution traffic, three features flagged
after the shift, and the two untouched features correctly stay silent. A
drift detector that fires constantly is worse than none, because it
teaches the on-call engineer to ignore it.

The shift here is **covariate shift**: a marketing push brings in newer
customers with more tickets and longer gaps between logins. The
relationship between features and churn is unchanged — only the input
distribution moved. Output monitoring alone can miss this; input
monitoring cannot.

The output moved too, and the project measures that as well: mean score
0.207 → 0.457, output PSI 1.16.

## 34.5 Rollback is a feature you test

```
before: v2-boosting p=0.0316
after : v1-logistic p=0.2565
matches v1 computed directly: True
```

A registry with two versions and a promote operation, exercised while
the service is running. The gate requires the rolled-back service to
reproduce v1's probability **exactly**, because a rollback that returns
"roughly the old model" is not a rollback.

Have one before you need one. The moment you need it is never the moment
to write it.

## 34.6 What this project does not cover

Honest limits, consistent with Appendix A. This is a single-process
service on loopback: no containers, no orchestration, no authentication,
no horizontal scaling, no CI/CD, no feature store, and no persistence of
the monitoring state between runs. *[PTG p.385–407]* covers local,
Heroku and Azure deployment and stops short of all of it too.

What the project *does* establish is the part that is usually skipped
and is not infrastructure at all: that the thing you serve computes what
you validated, that it refuses bad input, that you know its tail
latency, that you would notice the inputs changing, and that you can go
back.

---

# 35 · What the five have in common

1. **Gates declared before the run**, with the rule written out, so a
   reader can check the threshold against the chapter it came from.
2. **An audit pack** per project in `runs/<name>/report.json`.
3. **Honest failure.** Three of the five shipped with a failing gate on
   the first run. In all three the gate was wrong, and the correction is
   documented rather than quietly edited away.
4. **No downloads, no hidden state.** Data generated or bundled, seeds
   explicit, reruns reproduce.

The habit to take from this part is the third one. A failing check is a
claim about your understanding, and the useful question is never "how do
I make this pass?" but **"which of us is wrong — the code, or the
threshold I wrote down before I understood the problem?"**

Both answers are common. Only one of them is a bug.
