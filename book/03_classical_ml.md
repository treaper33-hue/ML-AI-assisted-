# Part II — Learning from Data

Part I built the engine. Part II is about everything that decides whether a
model is *true*: how the problem is framed, how the data is split, and how the
result is measured. Almost every failure in applied machine learning happens
here rather than in the model.

---

# 9 · Framing the problem

## 9.1 Decide the metric before the first fit

The most expensive mistakes are made before any code runs. Write these down,
in this order, *before* fitting anything *[SKL ch.1 "Choosing Metrics and
Thresholds That Match with the Goal"]*:

1. **The decision.** What will change because of this prediction? If nothing
   changes, stop.
2. **The cost of being wrong**, in both directions. A missed fraud and a
   blocked legitimate customer are not the same cost, and the ratio between
   them determines the threshold.
3. **The target.** Exactly which quantity, measured when, available at
   prediction time. ("Available at prediction time" is doing more work in that
   sentence than any other phrase in this book — see Chapter 10.)
4. **The metric**, chosen from the cost above, not from habit.
5. **The baseline.** Predict the mean, the majority class, or last week's
   value. If you cannot beat it, you have learned something important and
   cheap.

The reason for the ordering is brutal: once you have seen a score, you cannot
un-see it, and every subsequent choice of metric is contaminated by knowing
which one flatters you.

### Worked example — the threshold follows from the costs

Suppose a false negative (missed fraud) costs £500 and a false positive (a
blocked good customer) costs £5. Expected cost of alarming at probability p:

<div class="eq">alarm: 5·(1 − p) &nbsp;&nbsp;·&nbsp;&nbsp; stay silent: 500·p</div>

Alarm when 5(1 − p) < 500p, i.e. 5 < 505p, i.e. **p > 0.0099**. The optimal
threshold is 1%, not 50%. In general:

<div class="eq">p* = C<sub>FP</sub> ⁄ (C<sub>FP</sub> + C<sub>FN</sub>)</div>

The default 0.5 threshold is the special case where both errors cost the same
— which is almost never true, and is why the threshold is a *business*
parameter you tune on validation data, never a property of the model.

## 9.2 Problem types

| Type | Target | Typical metric |
|---|---|---|
| Binary classification | one of two labels | PR-AUC, F1, recall at fixed precision |
| Multi-class | one of k | macro-F1, balanced accuracy |
| Multi-label | any subset of k | per-label F1, subset accuracy |
| Regression | a real number | MAE, RMSE, R² |
| Forecasting | a future sequence | MASE, RMSSE (Chapter 23) |
| Ranking | an ordering | NDCG, MAP |
| Clustering | no target at all | silhouette *plus* stability |

## 9.3 Schema before model

Type every column before modelling *[SKL ch.3 "Foundations: Why Preprocessing
Matters and How to Read Data as a Schema"]*. Not the dtype pandas guessed —
the *logical* type, which determines the treatment:

| Logical type | Treatment |
|---|---|
| Continuous | scale; consider a transform if skewed |
| Count | often log1p; consider Poisson-style losses |
| Ordinal | integer encoding, order preserved |
| Nominal | one-hot, or target encoding fitted inside the fold |
| Datetime | decompose into cyclical parts; never feed a raw timestamp |
| Text | TF-IDF, then reduce (Chapter 15) |
| Identifier | **drop it** — and check it is not leaking (Chapter 10) |

A column whose logical type you cannot state is a column you do not
understand, and it will bite you later.

---

# 10 · Leakage

> Leakage is the use, at training time, of information that will not exist at
> prediction time. It does not announce itself. It appears as an excellent
> score.

This chapter takes its taxonomy from *[TSP p.158–p.164]*, the most rigorous
treatment in the sources, and applies it to *all* work, temporal or not.

## 10.1 The demonstration

Here is leakage with nothing else in the picture. The features are pure noise;
the target is independent of them. **There is no signal.** Any positive score
is manufactured.

```python
"""Feature selection outside the fold invents signal from pure noise."""

import numpy as np


def fit_predict(x_train, y_train, x_test):
    """Least-squares fit with an intercept, used as a neutral probe."""
    design = np.hstack([x_train, np.ones((x_train.shape[0], 1))])
    weights, *_ = np.linalg.lstsq(design, y_train, rcond=None)
    test_design = np.hstack([x_test, np.ones((x_test.shape[0], 1))])
    return test_design @ weights


def r2(y_true, y_pred):
    """Coefficient of determination."""
    total = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / total


rng = np.random.default_rng(20260101)
n, p, keep = 200, 2000, 20
x = rng.normal(size=(n, p))
y = rng.normal(size=n)                      # independent of x. No signal.

idx = rng.permutation(n)
train, test = idx[:140], idx[140:]

# LEAKING: rank features using every row, including the test rows.
leaked = np.argsort(np.abs(x.T @ (y - y.mean())))[-keep:]
leaked_r2 = r2(y[test], fit_predict(x[train][:, leaked], y[train],
                                    x[test][:, leaked]))

# HONEST: rank features using the training rows only.
honest = np.argsort(np.abs(x[train].T @ (y[train] - y[train].mean())))[-keep:]
honest_r2 = r2(y[test], fit_predict(x[train][:, honest], y[train],
                                    x[test][:, honest]))

print(f'leaked R² = {leaked_r2:+.3f}   honest R² = {honest_r2:+.3f}')
```

On the seed above this prints roughly **leaked R² = +0.35, honest R² = −0.39**.
A third of the variance "explained" in data that contains none.

### Why the inflation is exactly this large

The mechanism is quantifiable, which makes it much harder to dismiss. With p
independent noise features, the sample correlation of each with the target is
roughly N(0, 1/n). Selecting the top 20 of 2,000 is selecting the extreme
upper tail of 2,000 draws — and the maximum of p standard normals grows like
√(2 log p), which for p = 2000 is about 3.9 standard deviations.

Those features look predictive **on the rows used to select them**, which
here include the test rows. The test set is no longer independent of the model
— it participated in choosing the features — so it cannot measure
generalisation. The honest version selects on training rows only; the chosen
features look good there and, correctly, mean nothing on the test rows, giving
the slightly negative R² you would expect from fitting noise.

Run this before you trust any pipeline you have built. It is the cheapest
inoculation in this book.

## 10.2 The taxonomy

| Kind | Mechanism | Anchor |
|---|---|---|
| **Transformation** | A scaler, imputer or encoder fitted on all rows before splitting. Test statistics flow into training. | *[TSP p.159]* |
| **Selection** | Features chosen using the full dataset, as above. | *[SKL ch.9]* |
| **Lookforward** | Any use of future information: a shuffled split on a time series, a rolling statistic over the whole series, a label that post-dates the prediction. | *[TSP p.161]* |
| **Panel / group** | The same entity in both train and test — a patient, a customer, a machine. The model memorises the entity, not the pattern. | *[TSP p.163]* |
| **Target-derived** | A feature computed from the target, usually in disguise: a group aggregate, a target encoding fitted outside the fold. | *[TSP p.159]* |
| **Duplicate rows** | Near-duplicates straddling the split. | *[SKL ch.3]* |

## 10.3 The checklist

From *[TSP p.164]*, in the order that matters:

1. **Split first.** Before any transformation. Always.
2. Fit every parameterised transform — scalers, imputers, encoders, PCA,
   selectors — **inside** the fold, on training rows only.
3. Respect chronology where time exists.
4. Keep entities whole: group-aware splits for panel data.
5. For every feature, ask *when would this value be known?* If the answer is
   "after the thing I am predicting", drop it.
6. Keep the test set sealed until the very end — one look, one number.
7. Document the pipeline so a reviewer can check 1–6 without reading code.

## 10.4 The one defence that scales

Rules you must remember are rules you will eventually forget. Structure beats
vigilance: put the entire transformation chain inside a single estimator
object and let the cross-validation machinery do the splitting
*[SKL ch.3 "Putting It Together: One Leak-Proof ColumnTransformer"]*. Then
"fit on train only" is not a discipline you maintain — it is what the code
does.

---

# 11 · Pipelines that cannot leak

## 11.1 The shape

```python
"""One leak-proof estimator: preprocessing and model as a single object."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pipeline(numeric: list, categorical: list) -> Pipeline:
    """Builds a leak-proof preprocessing and modelling pipeline.

    Every parameterised step lives inside the pipeline, so it is fitted
    on training folds only when the pipeline is cross-validated.

    Args:
        numeric: Names of the continuous columns.
        categorical: Names of the nominal columns.

    Returns:
        An unfitted scikit-learn Pipeline.
    """
    numeric_steps = Pipeline([
        ('impute', SimpleImputer(strategy='median', add_indicator=True)),
        ('scale', StandardScaler()),
    ])
    categorical_steps = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('encode', OneHotEncoder(handle_unknown='ignore', min_frequency=10)),
    ])
    preprocess = ColumnTransformer([
        ('num', numeric_steps, numeric),
        ('cat', categorical_steps, categorical),
    ], remainder='drop')
    return Pipeline([
        ('prep', preprocess),
        ('model', HistGradientBoostingClassifier(random_state=0)),
    ])
```

Three details carry most of the value. `handle_unknown='ignore'` stops an
unseen category from crashing inference. `add_indicator=True` keeps the fact
that a value *was* missing, which is often more predictive than whatever you
impute. `remainder='drop'` means a new column cannot silently join the model
without you naming it.

## 11.2 Prove it

Do not take the pipeline's word for it. Measure the gap:

```python
from sklearn.model_selection import cross_val_score

honest = cross_val_score(build_pipeline(num_cols, cat_cols), X, y,
                         cv=5, scoring='average_precision')

# The wrong way, for contrast: transform everything up front, then CV.
X_all = preprocess.fit_transform(X)          # sees every row
leaky = cross_val_score(model, X_all, y, cv=5, scoring='average_precision')

print(f'honest {honest.mean():.4f}  leaky {leaky.mean():.4f}  '
      f'inflation {leaky.mean() - honest.mean():+.4f}')
```

Record that inflation number. It is the size of the lie you would have shipped.

## 11.3 Splitters

The pipeline protects you from transformation leakage. It does **not** protect
you from the other kinds — those are the splitter's job
*[SKL ch.3 "Leakage-Safe Evaluation"]*:

| Data | Splitter |
|---|---|
| i.i.d. rows | `KFold`, or `StratifiedKFold` when classes are imbalanced |
| Repeated entities | `GroupKFold`, `StratifiedGroupKFold` |
| Time | `TimeSeriesSplit`, or the rolling/expanding designs of Chapter 23 |
| Both group and time | a custom splitter; write the boundaries out and assert them |

---

# 12 · Linear models

## 12.1 The bias–variance decomposition

Before any particular model, the result that explains why regularisation can
*improve* predictions despite making the fit worse. Let y = f(x) + ε with
𝔼[ε] = 0, Var(ε) = σ², and let f̂ be our estimate (random, because the
training set is random). The expected squared error at a point x is

<div class="eq">𝔼[(y − f̂(x))²] = (f(x) − 𝔼[f̂(x)])² + 𝔼[(f̂(x) − 𝔼[f̂(x)])²] + σ²</div>

**Derivation.** Write y − f̂ = (y − f) + (f − 𝔼[f̂]) + (𝔼[f̂] − f̂). Square and
take expectations. The cross terms vanish: ε is independent of the training
data and has zero mean, and 𝔼[𝔼[f̂] − f̂] = 0 by construction. What survives
is the three terms above. ∎

<div class="eq">Error = Bias² + Variance + Irreducible noise</div>

Three consequences worth internalising:

- **σ² is a floor.** No model beats it; a reported error below it means a leak
  (Chapter 10) or a bug.
- **Bias and variance trade.** More capacity lowers bias and raises variance.
- **Regularisation buys variance reduction with bias.** It is a *deliberately
  worse fit* that generalises better — which is exactly what ridge does next.

## 12.2 Ridge, derived — and what α really does

Ordinary least squares minimises ‖X**w** − **y**‖². Ridge adds a penalty:

<div class="eq">L(<b>w</b>) = ‖X<b>w</b> − <b>y</b>‖² + α‖<b>w</b>‖²</div>

Differentiate — the same move as Chapter 4, with one extra term:

<div class="eq">∇L = 2Xᵀ(X<b>w</b> − <b>y</b>) + 2α<b>w</b> = 0 &nbsp;⟹&nbsp; (XᵀX + αI)<b>w</b> = Xᵀ<b>y</b></div>

### In the SVD basis, ridge becomes transparent

Write X = UΣVᵀ (Chapter 5). Substituting and simplifying:

<div class="eq"><b>w</b><sub>ridge</sub> = Σᵢ [ σᵢ ⁄ (σᵢ² + α) ] ⟨<b>u</b>ᵢ, <b>y</b>⟩ <b>v</b>ᵢ</div>

whereas OLS has 1/σᵢ in place of σᵢ/(σᵢ² + α). So ridge multiplies each
component by a **shrinkage factor**

<div class="eq">σᵢ² ⁄ (σᵢ² + α) ∈ (0, 1)</div>

Now read off the behaviour:

| Direction | Shrinkage factor | Effect |
|---|---|---|
| σᵢ² ≫ α (strong, well-determined) | ≈ 1 | barely touched |
| σᵢ² ≈ α | 0.5 | halved |
| σᵢ² ≪ α (weak, noise-dominated) | ≈ 0 | suppressed |

**Ridge does not shrink everything equally — it shrinks the directions your
data barely determines.** Those are precisely the directions where OLS divides
by a tiny σᵢ and amplifies noise, which is the κ(X) catastrophe of Chapter 4
appearing in statistical dress. The condition number of the ridge system is

<div class="eq">κ = (σ<sub>max</sub>² + α)⁄(σ<sub>min</sub>² + α) &lt; κ(XᵀX)</div>

so **regularisation is conditioning**, and the statistical story (variance
reduction) and the numerical story (stability) are the same story.

```python
import numpy as np


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Fits ridge regression in closed form, leaving the intercept free.

    Args:
        x: Feature matrix of shape (n, d).
        y: Targets of shape (n,).
        alpha: Non-negative penalty strength.

    Returns:
        Coefficients of shape (d + 1,), intercept last.

    Raises:
        ValueError: If alpha is negative.
    """
    if alpha < 0:
        raise ValueError('alpha must be non-negative')
    x_mean, y_mean = x.mean(axis=0), y.mean()
    xc, yc = x - x_mean, y - y_mean
    d = x.shape[1]
    w = np.linalg.solve(xc.T @ xc + alpha * np.eye(d), xc.T @ yc)
    return np.append(w, y_mean - x_mean @ w)
```

The intercept is excluded from the penalty deliberately: shrinking it would
tie predictions to the arbitrary location of your coordinate system.

## 12.3 L1 versus L2, finished properly

Lasso penalises ‖**w**‖₁. Chapter 2 noted the L1 ball has corners on the axes;
here is why that produces exact zeros, in one dimension where it is provable
by hand.

Minimise ½(w − c)² + α|w| — a least-squares term pulling w towards c, and an
L1 penalty pulling towards zero. For w > 0 the derivative is (w − c) + α,
zero at w = c − α. For w < 0 it is (w − c) − α, zero at w = c + α. And at
w = 0 the subgradient is [−α, α], so zero is optimal whenever |c| ≤ α. The
solution is the **soft-threshold**:

<div class="eq">w* = sign(c)·max(|c| − α, 0)</div>

**Exactly zero whenever |c| ≤ α** — a genuine selection, not a small number.

Now L2: minimise ½(w − c)² + αw². Differentiating gives w − c + 2αw = 0, so

<div class="eq">w* = c ⁄ (1 + 2α)</div>

which is zero only when c is. L2 shrinks proportionally and never selects.

> **L1 selects, L2 shrinks** — and now you know it is a statement about the
> subgradient at the origin, not folklore.

## 12.4 Logistic regression

For binary targets, model the log-odds as linear and squash with the sigmoid:

<div class="eq">p(<b>x</b>) = σ(<b>w</b>ᵀ<b>x</b> + b), &nbsp;&nbsp; σ(z) = 1 ⁄ (1 + e<sup>−z</sup>)</div>

Fit by maximum likelihood — which Chapter 8 showed is minimising binary
cross-entropy.

### The gradient, derived

Two facts make this short. First, σ′(z) = σ(z)(1 − σ(z)) — differentiate
(1 + e^{−z})^{−1} and regroup. Second, the per-sample loss is
−[y log p + (1−y) log(1−p)]. Differentiating with respect to z:

<div class="eq">∂L⁄∂z = −[ y⁄p − (1−y)⁄(1−p) ]·p(1−p) = −[ y(1−p) − (1−y)p ] = p − y</div>

The σ′ factor **cancels exactly** against the derivative of the log. Then by
the chain rule, since z = **w**ᵀ**x**:

<div class="eq">∂L⁄∂<b>w</b> = Xᵀ(<b>p</b> − <b>y</b>)</div>

*Prediction minus truth, projected back through the features.* ∎

The identical form appears in linear regression with squared error, and in the
softmax layer of Chapter 17. Not a coincidence: it is what happens whenever
the loss is the negative log-likelihood of an exponential-family distribution
with its matching link function.

### Why it is convex, and what Newton does

The Hessian is

<div class="eq">H = XᵀSX, &nbsp;&nbsp; S = diag(pᵢ(1 − pᵢ))</div>

Every pᵢ(1−pᵢ) > 0, so for any **v**, **v**ᵀH**v** = ‖S^{1/2}X**v**‖² ≥ 0.
**The Hessian is positive semi-definite everywhere, so the problem is convex**
and every local optimum is global — unlike anything in Part III.

Newton's method then gives the update

<div class="eq"><b>w</b> ← <b>w</b> + (XᵀSX)⁻¹Xᵀ(<b>y</b> − <b>p</b>)</div>

which is a weighted least-squares solve — hence the classical name
**iteratively reweighted least squares**, and the reason logistic regression
converges in a handful of iterations rather than the thousands gradient
descent would need.

Logistic regression is the strongest baseline in applied ML: fast, calibrated,
interpretable, convex. Beat it before you reach for anything deeper.

---

# 13 · Trees and ensembles

## 13.1 One tree, one split

A decision tree splits the feature space with axis-aligned cuts, choosing at
each node the split that most reduces impurity:

<div class="eq">Gini = 1 − Σ<sub>k</sub> p<sub>k</sub>² &nbsp;&nbsp;·&nbsp;&nbsp; Entropy = − Σ<sub>k</sub> p<sub>k</sub> log p<sub>k</sub></div>

That entropy is Chapter 8's entropy, and the "information gain" of a split is
the reduction in it.

### Worked example — choosing between two splits

Parent node: 100 samples, 50 positive. Gini = 1 − (0.5² + 0.5²) = **0.5**.

**Split A** → left (50: 40 pos), right (50: 10 pos):

<div class="eq">G<sub>left</sub> = 1 − (0.8² + 0.2²) = 0.32 &nbsp;·&nbsp; G<sub>right</sub> = 1 − (0.2² + 0.8²) = 0.32</div>
<div class="eq">weighted = 0.5·0.32 + 0.5·0.32 = 0.32 &nbsp;⟹&nbsp; gain = 0.5 − 0.32 = <b>0.18</b></div>

**Split B** → left (10: 10 pos), right (90: 40 pos):

<div class="eq">G<sub>left</sub> = 0 &nbsp;·&nbsp; G<sub>right</sub> = 1 − ((40⁄90)² + (50⁄90)²) ≈ 0.494</div>
<div class="eq">weighted = 0.1·0 + 0.9·0.494 ≈ 0.444 &nbsp;⟹&nbsp; gain ≈ <b>0.056</b></div>

Split A wins, despite Split B producing one perfectly pure leaf — because that
leaf is small, and the weighting by node size is what stops trees from chasing
tiny pure corners. Note also that the gain is **always non-negative**: a split
cannot increase weighted impurity, which is exactly why an unconstrained tree
grows until every leaf is pure, memorising the training set.

The depth, minimum-leaf-size and minimum-impurity-decrease parameters exist
entirely to stop that.

## 13.2 Bagging and boosting

| | Random forest | Gradient boosting |
|---|---|---|
| Trees are | independent, parallel | sequential, each fixing the last |
| Each tree is | deep, low-bias, high-variance | shallow, high-bias |
| Combination | average | weighted sum |
| Reduces | variance | bias |
| More trees | never hurts | eventually overfits |
| Tuning | forgiving | repays care |

### Why averaging works, and why correlation limits it

Average B estimators each with variance σ² and pairwise correlation ρ:

<div class="eq">Var(mean) = ρσ² + (1 − ρ)σ²⁄B</div>

As B → ∞ the second term vanishes but the first does not. **The floor is
ρσ².** Adding trees past a point buys nothing; the only way lower is to
*decorrelate* them — which is precisely what a random forest's random feature
subset at each split is for. Bagging alone (bootstrap samples, all features)
leaves highly correlated trees, and this formula is why Breiman added the
feature sampling.

**Boosting** instead fits each new tree to the *negative gradient of the loss*
with respect to the current predictions — Chapter 6's gradient descent,
performed in function space rather than parameter space. With squared error
the negative gradient is just the residual, which is why the classic
description is "fit the next tree to the residuals"; with other losses it is
not the residual, and the gradient view is the one that generalises.

For tabular data, gradient boosting is the honest default, and
`HistGradientBoostingClassifier` is fast enough to be the first thing you try.
Compare at *matched budgets*, though: a well-tuned forest beating a badly
tuned booster teaches nothing.

---

# 14 · Evaluation

## 14.1 The confusion matrix, worked

<div class="eq">Precision = TP⁄(TP+FP) &nbsp;·&nbsp; Recall = TP⁄(TP+FN) &nbsp;·&nbsp; F1 = 2·P·R⁄(P+R)</div>

Take 10,000 samples with a 1% positive rate, and a model that catches 80 of
the 100 positives while raising 400 false alarms:

| | Predicted + | Predicted − |
|---|---|---|
| **Actually +** | TP = 80 | FN = 20 |
| **Actually −** | FP = 400 | TN = 9,500 |

<div class="eq">Precision = 80⁄480 = <b>0.167</b> &nbsp;·&nbsp; Recall = 80⁄100 = <b>0.80</b> &nbsp;·&nbsp; F1 = <b>0.276</b></div>
<div class="eq">Accuracy = (80 + 9500)⁄10000 = <b>0.958</b></div>

**95.8% accuracy, and five out of six alarms are wrong.** Meanwhile the
do-nothing model that predicts "negative" always scores 99% accuracy. This is
Chapter 8's base-rate arithmetic wearing a different hat, and it is why
accuracy is banned from imbalanced reports.

Note too that F1 is the *harmonic* mean, which is why it sits much closer to
the smaller of precision and recall than the arithmetic mean would — it
refuses to let a good recall hide a terrible precision.

### PR-AUC versus ROC-AUC

ROC plots recall against the false-positive rate FP/(FP+TN). In the table
above, FPR = 400/9900 = 0.040 — a reassuring-looking number produced by 400
wrong alarms, because the denominator is the vast negative class. Precision
uses FP/(TP+FP) instead, where the positive class is in both terms, and so it
*feels* those 400.

**Rule:** when positives are rare, precision-recall is the honest curve
*[SKL ch.6]*. Report both; state which one decides.

### Macro versus micro

Macro averages the per-class scores (every class equal); micro pools the
counts (every *sample* equal). On imbalanced data they differ sharply — so
return both, always, or the reader cannot know which question you answered.

## 14.2 Regression metrics

<div class="eq">MAE = (1⁄n)Σ|yᵢ − ŷᵢ| &nbsp;·&nbsp; RMSE = √((1⁄n)Σ(yᵢ − ŷᵢ)²) &nbsp;·&nbsp; R² = 1 − SS<sub>res</sub>⁄SS<sub>tot</sub></div>

They are minimised by *different predictions*, and this is not a stylistic
matter. Minimising 𝔼|y − c| over c gives the **median**; minimising
𝔼(y − c)² gives the **mean**. So choosing MAE or RMSE is choosing what you
want the model to predict, and on skewed targets those differ substantially.

MAE ≤ RMSE always (Jensen), and the **gap between them is a free diagnostic**
for how heavy your error tail is: RMSE ≈ MAE means uniform errors; RMSE ≫ MAE
means a few large ones dominating.

Percentage errors (MAPE) are undefined at zero targets and are not scale
invariant; Chapter 23 shows why, and what to use instead.

## 14.3 Cross-validation and the optimism of selection

| Pattern | Use |
|---|---|
| k-fold | i.i.d. data; k = 5 or 10 |
| Stratified k-fold | imbalanced classification — the default |
| Group k-fold | repeated entities |
| Time-series split | anything temporal (Chapter 23) |
| Repeated k-fold | small data, to average away split noise |
| **Nested CV** | when you tune *and* report |

**Nested CV** deserves its own argument *[SKL ch.6 "Nested CV"]*. If you
select hyperparameters on the same folds you report, the reported number is
optimistically biased.

Quantify it. Suppose you evaluate m candidate configurations, all genuinely
equivalent, each measured with noise of standard deviation s. You report the
best. The expected maximum of m draws from N(0, s²) is approximately

<div class="eq">s·√(2 log m)</div>

so with m = 100 configurations and a CV standard deviation of s = 0.02, the
selection bias alone is about 0.02 × 3.03 ≈ **0.06** — six points of AUC
conjured from nothing. Nested CV puts the search in an inner loop and the
evaluation in an outer one, so the reported score never sees the selection.
The outer score is nearly always worse, and the gap *is* your optimism.
Measure it once; it will change how you read leaderboards forever.

## 14.4 Every number carries an interval

A single test score is a point estimate from a finite sample, with the √n
behaviour of Chapter 8. Report it alone and you invite the reader to believe
differences that are noise.

```python
import numpy as np


def bootstrap_ci(metric_fn, y_true, y_pred, n_boot=1000, level=0.95, seed=0):
    """Percentile bootstrap confidence interval for any metric.

    Args:
        metric_fn: Callable taking (y_true, y_pred) and returning a float.
        y_true: Ground-truth values.
        y_pred: Predicted values.
        n_boot: Number of resamples.
        level: Confidence level.
        seed: Seed for reproducibility.

    Returns:
        A (lower, upper) tuple.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    samples = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples[i] = metric_fn(y_true[idx], y_pred[idx])
    tail = (1.0 - level) / 2.0
    return (float(np.quantile(samples, tail)),
            float(np.quantile(samples, 1.0 - tail)))
```

The bootstrap works because the empirical distribution is a consistent
estimate of the true one, so resampling *from your sample* mimics resampling
from the population. It needs no distributional assumption, which is why it
applies to any metric — including ones with no closed-form standard error,
like AUC.

A useful sanity figure: for a proportion near 0.5, the 95% interval is about
±1/√n. With n = 400 that is ±5 points. **If two models differ by less than
that, you have not shown anything.**

## 14.5 Calibration

A model that outputs 0.8 should be right 80% of the time at that score. Many
are not: boosted trees are typically over-confident, SVMs badly scaled. If any
*decision* uses the probability — expected-value calculations like Section
9.1's threshold, triage, risk — calibration matters more than ranking quality.

Diagnose with a reliability curve (bin the predictions, plot mean predicted
against observed frequency) and summarise with the **Brier score**, the mean
squared error of the probabilities. It decomposes as

<div class="eq">Brier = calibration error − resolution + uncertainty</div>

which is why it rewards being both sharp *and* honest. Fix with Platt scaling
(a logistic fit on held-out scores) or isotonic regression (more flexible,
needs more data). Calibrate on data the model did not train on, or you have
simply moved the leak.

---

# 15 · Unsupervised structure

## 15.1 PCA in practice

Chapter 5 derived PCA three ways. What matters in use:

- **Standardise first** unless units are genuinely comparable — the variance
  being maximised has units.
- Choose k from the explained-variance curve, then **validate k downstream**:
  the right number of components is the one that helps the task, not the one
  at a visual elbow.
- Components are directions, not features with meaning, and the sign is
  arbitrary — never attach a story to it.
- PCA is linear. If the structure is curved it will miss it; kernel PCA or an
  autoencoder (Chapter 20) will not.

**t-SNE and UMAP are visualisation, not features** *[SKL ch.5]*. They do not
preserve global distances, their output changes with perplexity and seed, and
feeding their coordinates into a model is a documented error.

## 15.2 k-means, and why it always converges

k-means minimises within-cluster squared distance:

<div class="eq">J = Σ<sub>k</sub> Σ<sub><b>x</b> ∈ C<sub>k</sub></sub> ‖<b>x</b> − <b>μ</b><sub>k</sub>‖²</div>

alternating two steps. Both *provably* reduce J, which is the whole
convergence argument:

**Assignment step.** Each point moves to its nearest centroid. Since each
point's contribution is minimised independently and no other term changes, J
cannot increase.

**Update step.** Each centroid moves to its cluster's mean. For fixed
membership, J is a sum of ‖**x** − **μ**‖² terms whose minimiser over **μ** is
the mean — differentiate: Σ2(**μ** − **x**ᵢ) = 0 gives **μ** = mean. So again
J cannot increase.

J is non-increasing and bounded below by 0, and there are finitely many
possible assignments, so the algorithm terminates. ∎

It never guarantees the *global* optimum (hence `n_init`), and it carries
three assumptions you agree to whether you notice or not: clusters are
spherical, similarly sized, and separated by Euclidean distance. Violate any
one and the output will still look like clusters — that is the danger.

## 15.3 DBSCAN

Density-based: a point is *core* if it has `min_samples` neighbours within
`eps`; clusters grow from core points; the rest is noise. It finds arbitrary
shapes, decides k for itself, and reports outliers explicitly. The price is
sensitivity to `eps` and failure when cluster densities differ. Pick `eps`
from a k-distance plot (sort each point's distance to its k-th neighbour; take
the elbow) rather than by guessing.

## 15.4 How to know if a clustering is real

The **silhouette** of a point compares its mean distance a to its own cluster
with the mean distance b to the nearest other cluster:

<div class="eq">s = (b − a) ⁄ max(a, b) ∈ [−1, 1]</div>

Near 1 means well-placed; near 0 means on a boundary; negative means it
probably belongs elsewhere.

| Check | Meaning | Caveat |
|---|---|---|
| Silhouette | separation vs cohesion | **rewards spherical clusters** — flatters k-means, penalises DBSCAN |
| Davies–Bouldin | spread over separation; lower better | same geometric bias |
| **Stability** | re-run across seeds and bootstrap samples; do assignments agree? | the most honest of the three |
| Downstream use | does it improve a task you care about? | the only one that pays |

Always report stability across at least five seeds. A clustering that changes
when the seed does is a description of your random number generator.

---

# Part II gate

| Check | Threshold |
|---|---|
| Leakage demonstration reproduces | leaked R² exceeds honest by > 0.1 on pure noise |
| Pipeline vs transform-first inflation | measured and recorded |
| Your ridge against scikit-learn | coefficients agree to 1e-6 |
| Soft-threshold formula vs a lasso solver in 1-D | exact zeros below α |
| Your worked Gini split numbers (0.18 vs 0.056) | exact |
| Worked confusion matrix (P 0.167, R 0.80, F1 0.276) | exact |
| Nested vs flat CV | nested ≤ flat; optimism gap reported |
| Every headline number | carries a bootstrap interval |
| Clustering report | includes stability over ≥ 5 seeds |
