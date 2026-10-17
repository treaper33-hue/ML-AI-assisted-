# Worked Projects

Five complete, runnable projects — not snippets. Each one is a single
file, runs in seconds to a minute, generates or loads its own data
without downloading anything, declares its acceptance gates before it
starts, and writes an audit pack to `runs/`.

```bash
python run_all.py            # all four, with a summary
python p1_tabular_churn.py   # or one at a time
```

Each project needs the testbed environment (`cd ../testbed && .\setup.ps1`).
Where a dependency is missing a project reports `UNVERIFIED` and exits
rather than pretending to pass.

| # | Project | Chapters | Runtime | What it proves |
|---|---|---|---|---|
| **1** | [`p1_tabular_churn.py`](p1_tabular_churn.py) | 9–14 | ~40 s | A leak-proof classical pipeline, scored honestly |
| **2** | [`p2_digits_cnn.py`](p2_digits_cnn.py) | 16–19 | ~10 s | A CNN trained by hand, with every claim checked |
| **3** | [`p3_forecast_conformal.py`](p3_forecast_conformal.py) | 22–25 | ~2 s | Forecasting with intervals whose coverage is measured |
| **4** | [`p4_vae_elbo.py`](p4_vae_elbo.py) | 5, 8, 20 | ~20 s | A VAE whose two ELBO terms are visible |
| **5** | [`p5_serving_monitoring.py`](p5_serving_monitoring.py) | 28 | ~9 s | A real HTTP service, measured and monitored |

---

## 1 · Tabular classification

Generated churn data with the pathologies real data has: 18% positive
rate, missing-not-at-random values, an 18-level region column with a long
tail, and **repeated customers**, so rows are not independent.

The pipeline: schema card → group-aware split → prior baseline →
group-aware nested CV across logistic regression, random forest and
gradient boosting → isotonic calibration → cost-based threshold →
bootstrap intervals → slice analysis.

What to look at:

- **The threshold.** Costs of £5 (false alarm) and £120 (missed churn)
  put the optimum near 0.04, not 0.5. The run finds ≈0.033 empirically
  and the saving over a 0.5 threshold is printed in pounds.
- **The nested-vs-flat gap**, which is the optimism of tuning. In
  `--quick` mode it is near zero because the search is tiny; widen the
  grid and watch it grow. That is selection bias appearing in front of
  you.
- **The slice table**, ranked by PR-AUC rather than recall, because
  recall at a low threshold is near 1 everywhere and tells you nothing.

## 2 · A convolutional classifier, trained by hand

The bundled `load_digits` set (1,797 8×8 images, no download) and a small
CNN, with every mechanism exposed: He initialisation, an explicit
four-line training loop, a cosine schedule.

Checks, in order: a **gradient check** in double precision before any
training; an **LR range test** that measures where the loss diverges; a
**determinism proof** (a second run at the same seed must be identical);
a **checkpoint round-trip** that must reproduce logits *exactly*; then
accuracy, macro/micro F1, a confusion matrix, and temperature scaling.

What to look at:

- **The calibration result is a negative one.** The model is already
  well calibrated (ECE ≈ 0.014), so temperature fitted on 288 validation
  points improves validation ECE but moves test ECE by +0.0006 — well
  inside the ±1/√n ≈ 0.05 noise floor. The gates say exactly that rather
  than claiming an improvement.
- **Checkpoint equality is `0.00e+00`, not "close".** Anything else means
  your save/load path is lossy.

## 3 · Forecasting with real intervals

A daily series with trend, weekly and annual seasonality, autocorrelated
noise, and a **deliberate regime shift** at 75% through.

Diagnostics (decomposition, ACF, ADF *and* KPSS with both outcomes
recorded) → rolling-origin folds with printed boundaries → four
baselines and two models scored by MASE and RMSSE → a horizon curve →
split conformal intervals.

What to look at:

- **Coverage either side of the shift.** 94.6% before, 73.4% after, at a
  90% target. Exchangeability fails across the shift, so the guarantee
  fails with it. This is the headline result, and it is measured rather
  than assumed.
- **The baselines are not decoration.** Seasonal naive scores MASE ≈ 1.10
  while drift scores ≈ 3.10; a model that does not beat the best of them
  has not earned its place.
- **Width and Winkler score are printed next to coverage**, because
  coverage alone is trivially gamed by a wide interval.

## 4 · A VAE with the ELBO taken apart

A small VAE on the same digits, built so the two ELBO terms are never
summed out of sight.

What to look at:

- **The likelihood choice is the whole project.** With summed squared
  error over 64 pixels in [0,1], reconstruction is of order 5 while the
  KL is tens of nats — so the cheapest way to minimise the sum is to
  switch the latent off. That is posterior collapse caused by *unit
  mismatch*, and the first version of this project did exactly that. The
  Bernoulli likelihood is measured in nats, so the terms are
  commensurable and the model has a real trade-off to make. Pass
  `likelihood='gaussian'` to `elbo_terms` to reproduce the failure.
- **The beta sweep.** β = 0 leaves the KL unconstrained (≈65 nats, all 8
  units active); β = 20 collapses it to ≈0.0002 with nothing active. The
  summed loss looks unremarkable throughout. The split tells you which
  model you actually trained.
- **The fair baseline.** PCA is the MSE-*optimal* linear map, so the
  comparison autoencoder is trained on squared error too. It then beats
  PCA (1.43 vs 1.56); the β = 1 VAE is worse (2.84) because it is paying
  reconstruction for a latent you can sample from. That gap is a price,
  not a defect — and the gate says so.

---

## 5 · Serving a model, and watching it afterwards

Two model versions, a registry, and a **real HTTP service** on loopback —
not a mock. Real `urllib` requests, real sockets, measured latencies.

What to look at:

- **Training/serving skew is `0.00e+00`.** The whole fitted `Pipeline`
  is shipped, so there is no second implementation of the preprocessing
  to drift away from the first. This is the most common production
  failure and the defence is structural, not vigilance.
- **Every malformed request returns 400, never 500**: a missing field, a
  string where a number belongs, an infinity, a non-object.
- **p50 is 24 ms for a single row** — and almost none of that is the
  model. One-row `predict_proba` through a `ColumnTransformer` is
  dominated by pandas and framework overhead; batching moves this by an
  order of magnitude. Measure before you optimise the wrong layer.
- **Drift detection that does not cry wolf.** PSI flags 0 features on
  in-distribution traffic and 3 after a deliberate covariate shift
  (support tickets 7.22, days-since-login 6.67, tenure 1.19). The
  relationship between features and churn is unchanged — only the inputs
  moved — which is the case output monitoring alone can miss.
- **Rollback is exact.** After promoting v1, the service reproduces v1's
  probability to the last bit.

---

## What every project has in common

1. **Gates declared before the run**, each with the rule written out, so
   a reader can check the threshold against the chapter it came from.
2. **An audit pack** in `runs/<project>/report.json` — config, metrics,
   gates, environment — enough for someone else to check the work
   without rerunning it.
3. **Honest failure reporting.** Three of the five projects shipped with
   a failing gate on the first run. In every case the gate was wrong, not
   the code, and the fix is documented above rather than quietly
   rewritten.
4. **No downloads and no hidden state.** Data is generated or bundled;
   seeds are explicit; reruns reproduce.
