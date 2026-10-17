# Source Ledger — Extracted Algorithms, Models, and Test Parameters

> **Step 1 of the Prompt Guidelines:** *"Extract working algorithms, mathematical
> models, and test parameters. Be sure to specify page numbers and highlight gaps."*

This ledger is the single source of truth for *where* every claim in
[ML_LEARNING_PROGRAM.md](ML_LEARNING_PROGRAM.md) comes from. Nothing enters the
program without an anchor here. Nothing leaves the review stage without a status
tag.

---

## 0. Pagination basis (read this before citing anything)

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

## 1. [MML] Mathematics of Machine Learning — algorithm extraction

### Part I: Linear Algebra (p.1–262)

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

### Parts II–III: Calculus and Multivariable Calculus (p.265–484)

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

### Part IV: Probability (p.489–629)

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

## 2. [SKL] Prompting Scikit-Learn — workflow extraction

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

## 3. [PTG] PyTorch: The Practical Guide — implementation extraction

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

## 4. [TSP] Time Series with PyTorch — extraction (PDF page index)

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

## 5. [GPSG] Google Python Style Guide — the code contract

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

## 6. Gaps and blind spots requiring additional verification

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
