# Part IV — Sequences and Uncertainty

Temporal data punishes every evaluation sin immediately and without mercy. A
shuffled split on a time series will hand you a beautiful score and a useless
model, and nothing in the training loop will warn you. This part is therefore
as much about *measurement* as about architecture — which is why it is the
part that most separates a professional from a competent beginner.

---

# 22 · The structure of temporal data

## 22.1 What makes it different

Three properties break the i.i.d. assumption every method in Part II relies
on *[TSP ch.2]*:

- **Order carries information.** Shuffling destroys it.
- **Observations are correlated with their own past** — autocorrelation.
- **The generating process drifts.** Tomorrow may not be drawn from today's
  distribution.

The last one is the deepest. It means a model is always an extrapolation, and
that no amount of held-out data proves it will keep working.

## 22.2 Decomposition

A series is usefully read as level + trend + seasonality + remainder
*[TSP p.65]*. Two combination rules:

<div class="eq">additive: y<sub>t</sub> = T<sub>t</sub> + S<sub>t</sub> + R<sub>t</sub> &nbsp;&nbsp;·&nbsp;&nbsp; multiplicative: y<sub>t</sub> = T<sub>t</sub> × S<sub>t</sub> × R<sub>t</sub></div>

Use additive when the seasonal swing is a constant *size* *[TSP p.66]*,
multiplicative when it is a constant *proportion* *[TSP p.70]*.

### Worked example — which one, and the log that converts them

Two series, each with a December peak:

| Year | Series A (Dec − baseline) | Series B (Dec ÷ baseline) |
|---|---|---|
| 1 | 100 − 1000 = +100 | 1100 ÷ 1000 = 1.10 |
| 2 | 1200 − 1100 = +100 | 1210 ÷ 1100 = 1.10 |
| 3 | 1300 − 1200 = +100 | 1331 ÷ 1210 = 1.10 |

A is additive: the December bump is +100 regardless of level. B is
multiplicative: the bump is +10% and therefore grows in absolute terms. Take
logs of B:

<div class="eq">log(T·S·R) = log T + log S + log R</div>

and it becomes additive. **That is the entire reason logging is the first
thing done to a business series** — it converts proportional seasonality into
the constant kind, which every additive method can then handle.

STL *[TSP p.73]* does the decomposition robustly with LOESS, handles a
changing seasonal shape, and resists outliers. It is the sensible default.

## 22.3 ACF and PACF

The **autocorrelation function** is the correlation of the series with itself
at lag k *[TSP p.84]*:

<div class="eq">ρ(k) = Σ<sub>t=k+1</sub><sup>n</sup>(y<sub>t</sub> − ȳ)(y<sub>t−k</sub> − ȳ) ⁄ Σ<sub>t=1</sub><sup>n</sup>(y<sub>t</sub> − ȳ)²</div>

### Derivation — the ACF of an AR(1)

Knowing one ACF analytically makes every empirical plot readable. Let
yₜ = φyₜ₋₁ + εₜ with |φ| < 1 and white noise ε. The stationary variance
satisfies

<div class="eq">γ₀ = Var(y) = φ²γ₀ + σ² &nbsp;⟹&nbsp; γ₀ = σ²⁄(1 − φ²)</div>

For the covariance at lag k, multiply the recursion by yₜ₋ₖ and take
expectations. Since εₜ is independent of the past, that term drops:

<div class="eq">γ<sub>k</sub> = φγ<sub>k−1</sub> &nbsp;⟹&nbsp; γ<sub>k</sub> = φ<sup>k</sup>γ₀ &nbsp;⟹&nbsp; ρ(k) = φ<sup>k</sup></div>

∎ **The ACF of an AR(1) decays geometrically.** With φ = 0.7: ρ(1) = 0.70,
ρ(2) = 0.49, ρ(3) = 0.34 — exactly the decaying tail you see in real plots,
and the reason a slow decay suggests φ near 1, i.e. near non-stationarity.

The **partial** autocorrelation *[TSP p.86]* gives the correlation at lag k
with the intervening lags' effects removed. For that same AR(1), PACF(1) = φ
and PACF(k) = 0 for k > 1 — all of the lag-2 correlation was *inherited*
through lag 1, none of it direct. That contrast is what the two plots are for:

| Pattern | Reading |
|---|---|
| ACF decays slowly, almost linearly | non-stationary; difference it |
| ACF decays geometrically, PACF cuts off after p | autoregressive, order p |
| ACF cuts off after q, PACF decays | moving average, order q |
| ACF spikes at 7, 14, 21 | weekly seasonality |

For a neural model you are not fitting ARIMA, but these plots still tell you
*which lags to feed it* — the cheapest feature engineering in forecasting.

```python
def acf(series: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocorrelation up to max_lag.

    Args:
        series: One-dimensional series.
        max_lag: Highest lag to compute.

    Returns:
        Array of length max_lag + 1, starting at lag 0 (always 1.0).
    """
    y = np.asarray(series, dtype=np.float64)
    y = y - y.mean()
    denom = float(y @ y)
    return np.array([float(y[k:] @ y[:len(y) - k]) / denom
                     for k in range(max_lag + 1)])
```

## 22.4 Stationarity

**Weak stationarity** means the mean is constant, the variance is finite and
constant, and the covariance γ(k) depends only on the lag k, never on where in
the series you are *[TSP p.92]*. Most statistical methods assume it; most real
series lack it.

Three tests, and they do not agree by construction *[TSP p.97]*:

| Test | Null hypothesis | Rejecting means |
|---|---|---|
| **ADF** | has a unit root (non-stationary) | stationary |
| **KPSS** | is stationary | non-stationary |
| **PP** | has a unit root, robust to heteroskedasticity | stationary |

ADF and KPSS have *opposite* nulls, so all four combinations occur:

| ADF | KPSS | Reading |
|---|---|---|
| rejects | fails to reject | stationary — both agree |
| fails to reject | rejects | non-stationary — both agree |
| fails to reject | fails to reject | **not enough data to decide** |
| rejects | rejects | likely trend-stationary, or heteroskedastic |

**Record both outcomes.** Choosing whichever test agrees with your plan is the
most common quiet dishonesty in applied forecasting, and it is invisible in
the final report unless you make it visible.

### What differencing does

Fix non-stationarity by differencing, Δyₜ = yₜ − yₜ₋₁. On a random walk
(φ = 1, the unit-root case) differencing returns exactly the white-noise
innovations — which is why a random walk's best forecast is its last value,
the naive baseline of Section 23.3.

Beware **over-differencing**: applying Δ to series that is already stationary
introduces a negative lag-1 autocorrelation of −0.5 that was not there before,
and your diagnostics will now show structure you created. Difference the
minimum number of times the tests require, and no more.

## 22.5 Structure determines the splitter

*[TSP p.103–106]* distinguishes four shapes, and getting this wrong
guarantees the wrong evaluation:

| Structure | Example | Splitter |
|---|---|---|
| Univariate | one store's daily sales | fixed origin / rolling |
| Multivariate | several series predicted jointly | as above, all aligned |
| Multivariable | one target, exogenous drivers | as above, drivers lagged |
| **Panel** | 10,000 stores × 3 years | **time-aware *and* group-aware** |

---

# 23 · Honest temporal evaluation

## 23.1 Never shuffle

A random split lets the model interpolate between points it has already seen,
which is not the task. Production requires extrapolation past the end of the
data. Three valid designs:

**Fixed origin** *[TSP p.126]* — train on everything up to T, test after.
Simple, one estimate, high variance.

**Expanding window** *[TSP p.132]* — the origin moves forward and training
history grows. Uses all data, mirrors a model retrained periodically.

**Rolling window** *[TSP p.134]* — fixed-length history slides forward.
Constant training size across folds, so folds are comparable, and it adapts if
the process drifts.

### Worked fold boundaries

n = 120, four folds, horizon 10, window 50:

| Fold | Train | Test |
|---|---|---|
| 0 | [0, 50) | [50, 60) |
| 1 | [10, 60) | [60, 70) |
| 2 | [20, 70) | [70, 80) |
| 3 | [30, 80) | [80, 90) |

Every training window is exactly 50 long, every test block strictly follows
its training block, and the last index used is 90 — so you need n ≥ window +
folds × horizon = 90, and the remaining 30 points are available for a final
held-out test. Compare with the expanding design, where fold 3 would train on
[0, 80): more data, but fold 0 and fold 3 are no longer comparable because
they saw different amounts of history.

```python
def rolling_window_cv(n: int, n_folds: int, horizon: int, window: int) -> list:
    """Builds rolling-origin folds whose training length is constant.

    Args:
        n: Length of the series.
        n_folds: Number of folds.
        horizon: Test length per fold.
        window: Fixed training-window length.

    Returns:
        A list of (train_idx, test_idx, description) tuples.

    Raises:
        ValueError: If the series is too short for the design.
    """
    needed = window + n_folds * horizon
    if needed > n:
        raise ValueError(f'need {needed} points, have {n}')
    folds = []
    for i in range(n_folds):
        end = window + i * horizon
        folds.append((np.arange(end - window, end),
                      np.arange(end, end + horizon),
                      f'train [{end - window},{end}) < test [{end},{end + horizon})'))
    return folds


def assert_no_lookforward(folds: list) -> None:
    """Fails if any training index reaches past its test window.

    Raises:
        AssertionError: If a fold leaks the future into training.
    """
    for train_idx, test_idx, label in folds:
        if train_idx.max() >= test_idx.min():
            raise AssertionError(f'lookforward leak in {label}')
```

Print the boundaries into your report and assert them in code. A reviewer who
cannot see them cannot check the evaluation, and an assertion that runs is
worth more than a convention you intend to follow.

## 23.2 Metrics, and the two traps

Start with the ordinary ones *[TSP p.138–144]*: MAE, MSE, RMSE as in
Chapter 14. Then the two failures specific to forecasting.

**Trap 1 — percentage errors break at zero.** APE = |y − ŷ|/|y| is undefined
when y = 0 *[TSP p.144]*. Intermittent demand is full of zeros. An
implementation that returns infinity poisons every aggregate downstream in
silence; the right behaviour is to refuse.

**Trap 2 — MAPE is not scale invariant.** *[TSP p.143]* makes this vivid with
a temperature table. Reproduce it: a forecast errs by 1° on a true value of
20 °C.

<div class="eq">MAPE in Celsius = 1⁄20 = <b>5.0%</b> &nbsp;&nbsp;·&nbsp;&nbsp; MAPE in Kelvin = 1⁄293.15 = <b>0.34%</b></div>

Same model, same error, same physical reality — and a fifteenfold difference
in the reported metric, produced entirely by the choice of zero point. Any
metric that behaves this way cannot compare series.

The fix is to scale by a **naive forecast computed on the training data**
*[TSP p.150]*:

<div class="eq">MASE = MAE(model) ⁄ MAE(naive on train) &nbsp;&nbsp;·&nbsp;&nbsp; RMSSE = RMSE(model) ⁄ RMSE(naive on train)</div>

### Worked MASE

Training series: 10, 12, 11, 13, 12, 14. The one-step naive differences are
|12−10|, |11−12|, |13−11|, |12−13|, |14−12| = 2, 1, 2, 1, 2, so

<div class="eq">naive scale = (2+1+2+1+2)⁄5 = <b>1.6</b></div>

Now test: truth 15, 14, 16; forecast 14, 15, 15. Absolute errors 1, 1, 1, so
MAE = 1.0 and

<div class="eq">MASE = 1.0 ⁄ 1.6 = <b>0.625</b></div>

Below 1, so the model beats the naive forecast — by 37.5%. Add 273.15 to every
number and recompute: the differences are unchanged, the errors are unchanged,
and MASE is **still 0.625**. That invariance is why it is this text's primary
forecasting metric.

> **The denominator uses training data only.** Computing the naive baseline on
> the test window is a leak, and an easy one to commit because the code looks
> symmetric.

```python
def naive_scale(y_train: np.ndarray, seasonality: int = 1) -> float:
    """MAE of the seasonal naive forecast on the TRAINING series.

    Args:
        y_train: Training portion, in time order.
        seasonality: Seasonal period; 1 is the random walk.

    Returns:
        The scaling constant for MASE.

    Raises:
        ValueError: If the series is too short or constant.
    """
    y = np.asarray(y_train, dtype=np.float64)
    if y.size <= seasonality:
        raise ValueError('training series too short for this seasonality')
    scale = float(np.mean(np.abs(y[seasonality:] - y[:-seasonality])))
    if scale == 0.0:
        raise ValueError('constant training series; MASE undefined')
    return scale


def mase(y_true, y_pred, y_train, seasonality: int = 1) -> float:
    """Mean absolute scaled error."""
    mae = float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))
    return mae / naive_scale(y_train, seasonality)
```

## 23.3 Baselines you must beat

| Baseline | Prediction | Optimal when |
|---|---|---|
| Naive | ŷₜ₊ₕ = yₜ | the series is a random walk |
| Seasonal naive | ŷₜ₊ₕ = yₜ₊ₕ₋ₛ | strong stable seasonality |
| Drift | last value + average slope | random walk with drift |
| Mean | the training mean | the series is stationary noise |

The first row is not a joke. Section 22.4 showed that differencing a random
walk leaves white noise — meaning *nothing* is predictable beyond the last
value, and the naive forecast is provably optimal. On short or noisy series
these baselines frequently win.

**Report it when they do.** A neural network losing to seasonal naive is a
finding about the problem — the signal is weak, or the horizon too long — and
suppressing it is the single most common dishonesty in applied forecasting.

## 23.4 Horizon

Error grows with horizon, and a single averaged number hides the shape
*[TSP p.156]*. For an AR(1), the h-step forecast is φʰyₜ and its error
variance is

<div class="eq">σ²·(1 + φ² + φ⁴ + ⋯ + φ<sup>2(h−1)</sup>) = σ²·(1 − φ<sup>2h</sup>)⁄(1 − φ²)</div>

which rises to the unconditional variance σ²/(1−φ²) as h grows. **Beyond some
horizon, your model knows nothing that the series' long-run mean does not** —
and that horizon is a property of the data, not of your architecture. Plot
metric against h; a model excellent at h = 1 and useless at h = 7 is a
different product from one that degrades gently.

---

# 24 · Neural forecasting

## 24.1 Windowing

Supervised learning needs (X, y) pairs, so slide a window: inputs
y_{t−L+1..t}, target y_{t+1..t+h}. Two rules:

1. **Build windows inside the fold**, after splitting. A window that straddles
   the boundary puts future values into a training input *[TSP p.161]*.
2. **Scale with training statistics only**, and invert before computing
   metrics.

Rule 1 has a quiet corollary: with a window length L, the first L−1 points of
each test block cannot be predicted without reaching back into training data.
That is legitimate (the past is known at prediction time) but it means your
effective test size is smaller than it looks, and the folds overlap in inputs
even when they do not overlap in targets.

## 24.2 Recurrent networks

An RNN carries a hidden state forward *[TSP ch.8 p.351]*:

<div class="eq">h<sub>t</sub> = tanh(W<sub>h</sub>h<sub>t−1</sub> + W<sub>x</sub>x<sub>t</sub> + b)</div>

### Why gradients vanish or explode, exactly

Backpropagation through time unrolls this into a network whose depth is the
sequence length. The gradient of a loss at time T with respect to a state at
time t involves the product

<div class="eq">∂h<sub>T</sub>⁄∂h<sub>t</sub> = Π<sub>k=t+1</sub><sup>T</sup> ∂h<sub>k</sub>⁄∂h<sub>k−1</sub> = Π<sub>k</sub> diag(1 − h<sub>k</sub>²)·W<sub>h</sub></div>

Bound the norm: ‖∂h_T/∂h_t‖ ≤ (‖W_h‖·max|tanh′|)^{T−t}. Since |tanh′| ≤ 1, if
the largest singular value of W_h is below 1 the product **decays
geometrically** in T − t; if it is above 1 it **explodes** geometrically. ∎

Long dependencies are therefore unlearnable in a plain RNN except in the
knife-edge case where the spectral radius sits at exactly 1.

**LSTM** solves it with gates and an additive cell state:

<div class="eq">f<sub>t</sub> = σ(W<sub>f</sub>[h<sub>t−1</sub>, x<sub>t</sub>]) &nbsp;·&nbsp; i<sub>t</sub> = σ(W<sub>i</sub>[h<sub>t−1</sub>, x<sub>t</sub>]) &nbsp;·&nbsp; o<sub>t</sub> = σ(W<sub>o</sub>[h<sub>t−1</sub>, x<sub>t</sub>])</div>
<div class="eq">c<sub>t</sub> = f<sub>t</sub> ⊙ c<sub>t−1</sub> + i<sub>t</sub> ⊙ tanh(W<sub>c</sub>[h<sub>t−1</sub>, x<sub>t</sub>]) &nbsp;·&nbsp; h<sub>t</sub> = o<sub>t</sub> ⊙ tanh(c<sub>t</sub>)</div>

The decisive line is the cell update. Its derivative along the cell path is

<div class="eq">∂c<sub>t</sub>⁄∂c<sub>t−1</sub> = f<sub>t</sub></div>

a *gate value*, not a weight matrix times an activation derivative. With the
forget gate near 1, the gradient passes through essentially unattenuated for
hundreds of steps — the **constant error carousel**. It is the same additive
trick as the residual connection of Chapter 21, discovered two decades
earlier.

`papers/02_Understanding_LSTM` implements all four gates from scratch; the GRU
merges them into two and is usually as good with fewer parameters.

Always **clip gradients** when training recurrent models — `clip_grad_norm_`
at 1.0 is standard and prevents a single exploding batch from destroying the
run. Clipping addresses the explosion half of the analysis above; the gates
address the vanishing half.

## 24.3 Transformers for forecasting

Attention (Chapter 21) removes the sequential bottleneck: every position sees
every other in one step, so a long dependency is one hop away rather than L.
The trade is explicit:

| | RNN / LSTM | Transformer |
|---|---|---|
| Path length between distant positions | O(L) | **O(1)** |
| Compute per layer | O(L·d²) | O(L²·d) |
| Parallel over time? | no | **yes** |
| Inductive bias for order | built in | **must be supplied** |

That last row is why positional encoding is mandatory, and why transformers
need more data than an LSTM to reach the same place on small series — they
must *learn* what recurrence assumes.

Use a **causal mask** so position t attends only to ≤ t. Omitting it is
lookforward leakage implemented in matrix form, and it produces spectacular
validation numbers that collapse in production.

**Benchmark honestly:** same folds, same scaler, same metric, matched
parameter and compute budgets. Transformers do not automatically beat an LSTM
on a few thousand points, and published comparisons that say otherwise usually
differ in budget.

---

# 25 · Uncertainty: conformal prediction

## 25.1 Why a point forecast is not enough

"Demand will be 1,000" supports no decision. "Demand will be between 900 and
1,100 with 90% probability" supports inventory, staffing and risk. Most
intervals in practice come from distributional assumptions that are wrong —
Gaussian residuals on heavy-tailed data — and are therefore too narrow exactly
when they matter *[TSP p.311]*.

**Conformal prediction** gives intervals with a finite-sample coverage
guarantee and almost no assumptions: only **exchangeability**, meaning the
joint distribution is invariant to reordering. (Weaker than i.i.d., and
notably *not* satisfied by drifting time series — see 25.4.)

## 25.2 The mechanism

The idea is a ranking, not a distribution *[TSP p.316–317]*. Define a
**non-conformity score** measuring how badly a point fits the model —
absolute error is the usual choice. Score every point in a held-out
calibration set. A new prediction's plausible range is then bounded by a
quantile of those scores.

**Split conformal**, in full:

1. Split into train / calibration / test.
2. Fit the model on train only.
3. Score residuals sᵢ = |yᵢ − ŷᵢ| on calibration.
4. Take q = the ⌈(n+1)(1−α)⌉-th smallest of those scores.
5. Predict ŷ ± q.

### Why coverage is guaranteed — the rank argument

This is the whole theoretical content, and it is one paragraph.

Under exchangeability, the test score s_{n+1} and the calibration scores
s₁…sₙ are exchangeable, so **s_{n+1} is equally likely to occupy any of the
n+1 possible ranks** in the combined set. The interval fails to cover exactly
when s_{n+1} exceeds q, i.e. when its rank is above ⌈(n+1)(1−α)⌉. The number
of such ranks is at most α(n+1), so

<div class="eq">P(miscoverage) ≤ α(n+1)⁄(n+1) = α &nbsp;⟹&nbsp; P(coverage) ≥ 1 − α</div>

∎ No distributional assumption appears anywhere: the guarantee is purely
combinatorial, which is why the (n+1) correction is not a fudge but the exact
count of available ranks.

### Worked example

With n = 99 calibration residuals and α = 0.1:

<div class="eq">⌈(99+1)(1 − 0.1)⌉ = ⌈90⌉ = 90</div>

so q is the 90th smallest residual. Nine of the 99 calibration points lie
above it, and the guarantee says a fresh point exceeds it with probability at
most 10%.

```python
def split_conformal(residuals: np.ndarray, alpha: float = 0.1) -> float:
    """Returns the conformal half-width for 1 - alpha coverage.

    Args:
        residuals: Absolute errors on the calibration set.
        alpha: Miscoverage rate; 0.1 targets 90% coverage.

    Returns:
        The half-width q to add and subtract from a point prediction.

    Raises:
        ValueError: If the calibration set is too small for this alpha.
    """
    s = np.sort(np.asarray(residuals, dtype=np.float64))
    n = s.size
    if n < int(np.ceil(1.0 / alpha)) - 1:
        raise ValueError(f'need at least {int(np.ceil(1/alpha)) - 1} '
                         f'calibration points for alpha={alpha}')
    k = int(np.ceil((n + 1) * (1.0 - alpha))) - 1
    return float(s[min(k, n - 1)])


def coverage(y_true, y_lower, y_upper) -> float:
    """Fraction of truths that fall inside their interval."""
    y_true = np.asarray(y_true)
    return float(np.mean((y_true >= y_lower) & (y_true <= y_upper)))
```

Note the minimum calibration size the code enforces: with α = 0.05 you need at
least 19 points, because ⌈(n+1)·0.95⌉ must not exceed n. Ask for 99% coverage
from 50 calibration points and the quantile does not exist — the honest answer
is an infinite interval, so the function refuses.

## 25.3 Adaptive intervals

Plain split conformal gives a **constant** half-width q: the same interval
everywhere, regardless of whether a point is in a well-understood region or an
extrapolation. Often that is not what you want.

**Normalised scores** fix it. Use sᵢ = |yᵢ − ŷᵢ|/ûᵢ where û is a second model
predicting the local error magnitude. The interval becomes ŷ ± q·û(x) — wider
where the model expects to struggle — and the coverage proof is unchanged,
because the rank argument never looked at how the score was defined.

**Conformalised quantile regression** goes further: fit models for the α/2 and
1−α/2 quantiles directly, then conformalise the residuals of those bounds.

## 25.4 The honest caveats

**Exchangeability fails under drift.** Time series are the canonical
violation: tomorrow is not exchangeable with last year. The guarantee degrades
exactly when the process changes — which is when you needed it. So measure
empirical coverage on a held-out *future* window, not only in-sample, and
report it.

**The literature's upper bound may not hold** *[TSP p.332]*. Carry that
warning into your own reports rather than quoting a guarantee you have not
tested.

**Coverage without width is meaningless.** An interval from −∞ to +∞ has
perfect coverage. Always report both, plus a proper scoring rule such as
pinball loss or the Winkler score *[TSP p.346]*. The Winkler score for an
interval [l, u] is

<div class="eq">(u − l) + (2⁄α)·(l − y) if y &lt; l, &nbsp; (u − l) + (2⁄α)·(y − u) if y &gt; u, &nbsp; else (u − l)</div>

— width, plus a penalty proportional to how far outside the truth fell. It is
minimised by an interval that is both narrow *and* correctly covering, which
is exactly the trade you want scored.

For time series, **EnbPI** *[TSP p.337]* replaces the single calibration split
with bootstrap ensembles:

1. Draw B bootstrap samples of the training series; fit a model on each.
2. For each training point, aggregate the predictions of only those models
   that did **not** see it — a leave-one-out residual at no extra cost.
3. Form intervals from the quantiles of those residuals, sliding the residual
   window forward as new observations arrive.

This keeps more data in training and adapts to drift. *[TSP p.340]* applies it
to a PyTorch model directly.

## 25.5 Anomaly detection, briefly

Conformal intervals give anomaly detection for free: a point outside its
interval is non-conforming at level α *[TSP ch.17 p.690]*. Other approaches —
isolation forests, autoencoder reconstruction error, Matrix Profile — differ
mainly in what "normal" means.

The hard part is never the algorithm; it is evaluation. Labels are scarce,
subjective and usually retrospective. Report precision@k against whatever
labels you have, say exactly how they were produced, and resist any claim that
rests on a plot looking right.

---

# Part IV gate

| Check | Threshold |
|---|---|
| Your ACF of a simulated AR(0.7) | ρ(k) ≈ 0.7ᵏ within sampling error |
| Your ACF/PACF vs `statsmodels` | 1e-8 |
| ADF, KPSS and PP all run | both outcomes recorded when they disagree |
| Fold boundaries match the worked table | exact |
| MASE worked example | 0.625 exactly |
| MAPE under a unit change | changes 15-fold; MASE unchanged |
| Neural model vs seasonal naive | reported either way, including a loss |
| Conformal coverage at α = 0.2, 0.1, 0.05 | within 2% of target, width reported |
| Coverage under a deliberate shift | measured, not assumed |
