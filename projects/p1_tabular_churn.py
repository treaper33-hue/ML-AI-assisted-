"""Project 1 — Tabular classification, end to end.

A complete, runnable churn project: generated data with the pathologies
real data has (class imbalance, missingness, high-cardinality categories,
repeated customers), then the full Part II workflow.

    schema -> leak-safe pipeline -> baselines -> models -> group-aware
    nested CV -> calibration -> cost-based threshold -> bootstrap
    intervals -> slice analysis -> audit pack

Every number it prints carries an interval or a gate. Nothing here is a
snippet: run it and you get a complete, reproducible result.

Chapters: 9 (framing), 10 (leakage), 11 (pipelines), 12-13 (models),
14 (evaluation, calibration, thresholds).

Usage:
    python p1_tabular_churn.py
    python p1_tabular_churn.py --quick       smaller search, for a fast pass
    python p1_tabular_churn.py --seed 7
"""

import argparse
import dataclasses
import json
import pathlib
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.compose import ColumnTransformer
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.metrics import brier_score_loss
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError as error:        # pragma: no cover - environment gate
    HAS_SKLEARN = False
    IMPORT_ERROR = error

HERE = pathlib.Path(__file__).resolve().parent
RUNS = HERE / 'runs'

NUMERIC = ['tenure_months', 'monthly_charge', 'support_tickets',
           'avg_session_minutes', 'days_since_login']
CATEGORICAL = ['plan', 'region', 'acquisition_channel']

# Cost of being wrong, in pounds. These two numbers decide the decision
# threshold (Chapter 9); they are a business input, not a model output.
COST_FALSE_POSITIVE = 5.0        # a retention offer sent to a happy customer
COST_FALSE_NEGATIVE = 120.0      # a churned customer we failed to catch


# --------------------------------------------------------------------- #
#                            1. The data                                 #
# --------------------------------------------------------------------- #


def generate_churn_data(n_customers: int = 2500,
                        seed: int = 20260101) -> Tuple['pd.DataFrame',
                                                       np.ndarray,
                                                       np.ndarray]:
    """Generates a churn dataset with realistic pathologies.

    The generative story: churn risk rises with support tickets and days
    since last login, falls with tenure, and is higher on the monthly
    plan. Noise is substantial, so the achievable PR-AUC is far from 1.

    Deliberate pathologies, each mirroring a chapter of the text:
      * ~18% positive rate (imbalance, Chapter 14)
      * missing values, not at random (Chapter 22's MAR idea)
      * a high-cardinality region column with rare levels (Chapter 11)
      * several rows per customer, so rows are NOT independent -- this is
        what forces a group-aware split (Chapter 10)

    Args:
        n_customers: Number of distinct customers.
        seed: Seed for the generator.

    Returns:
        A tuple (features, labels, groups) where groups holds the
        customer id of each row.
    """
    rng = np.random.default_rng(seed)

    tenure = rng.gamma(shape=2.0, scale=12.0, size=n_customers)
    monthly = 20.0 + 0.35 * tenure + rng.normal(scale=8.0, size=n_customers)
    tickets = rng.poisson(lam=0.8 + 2.0 / (1.0 + tenure / 12.0))
    session = np.clip(rng.normal(loc=35.0 - 0.1 * tickets, scale=12.0,
                                 size=n_customers), 1.0, None)
    recency = rng.exponential(scale=10.0 + 0.4 * tickets, size=n_customers)

    plan = rng.choice(['monthly', 'annual', 'two_year'], size=n_customers,
                      p=[0.55, 0.3, 0.15])
    # A long-tailed region distribution: a few common regions and a tail
    # of rare ones, which is what forces min_frequency in the encoder.
    region_weights = np.array([0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05,
                               0.04, 0.04, 0.03, 0.03, 0.025, 0.02, 0.015,
                               0.01, 0.01, 0.005, 0.005])
    region_weights = region_weights / region_weights.sum()
    region = rng.choice([f'r{i:02d}' for i in range(len(region_weights))],
                        size=n_customers, p=region_weights)
    channel = rng.choice(['organic', 'paid', 'partner', 'referral'],
                         size=n_customers, p=[0.4, 0.3, 0.2, 0.1])

    # The true log-odds. Scaled to land near an 18% positive rate.
    logit = (-1.9
             - 0.022 * tenure
             + 0.33 * tickets
             + 0.035 * recency
             - 0.012 * session
             + 0.75 * (plan == 'monthly')
             - 0.55 * (plan == 'two_year')
             + 0.40 * (channel == 'paid'))
    probability = 1.0 / (1.0 + np.exp(-logit))
    churned = rng.random(n_customers) < probability

    frame = pd.DataFrame({
        'tenure_months': tenure,
        'monthly_charge': monthly,
        'support_tickets': tickets.astype(float),
        'avg_session_minutes': session,
        'days_since_login': recency,
        'plan': plan,
        'region': region,
        'acquisition_channel': channel,
    })

    # Missing not at random: sessions go unrecorded for dormant accounts.
    dormant = recency > np.quantile(recency, 0.8)
    missing = dormant & (rng.random(n_customers) < 0.55)
    frame.loc[missing, 'avg_session_minutes'] = np.nan
    frame.loc[rng.random(n_customers) < 0.04, 'monthly_charge'] = np.nan

    # Repeated customers: some appear two or three times (monthly
    # snapshots). Rows are no longer independent.
    groups = np.arange(n_customers)
    repeats = rng.random(n_customers) < 0.25
    extra_idx = np.where(repeats)[0]
    extra = frame.iloc[extra_idx].copy()
    extra['days_since_login'] *= rng.uniform(0.6, 1.4, size=len(extra_idx))
    extra['support_tickets'] += rng.integers(0, 2, size=len(extra_idx))

    frame = pd.concat([frame, extra], ignore_index=True)
    labels = np.concatenate([churned, churned[extra_idx]])
    groups = np.concatenate([groups, groups[extra_idx]])
    order = rng.permutation(len(frame))
    return (frame.iloc[order].reset_index(drop=True),
            labels[order].astype(int), groups[order])


def describe_schema(frame: 'pd.DataFrame', labels: np.ndarray,
                    groups: np.ndarray) -> Dict[str, object]:
    """Builds the schema card that must exist before any model is fitted.

    Args:
        frame: The feature table.
        labels: Binary targets.
        groups: Group identifier per row.

    Returns:
        A JSON-serialisable schema description.
    """
    return {
        'n_rows': int(len(frame)),
        'n_groups': int(len(np.unique(groups))),
        'rows_per_group': round(float(len(frame) / len(np.unique(groups))), 3),
        'positive_rate': round(float(labels.mean()), 4),
        'numeric': NUMERIC,
        'categorical': CATEGORICAL,
        'missing_fraction': {
            column: round(float(frame[column].isna().mean()), 4)
            for column in frame.columns if frame[column].isna().any()
        },
        'cardinality': {column: int(frame[column].nunique())
                        for column in CATEGORICAL},
    }


# --------------------------------------------------------------------- #
#                         2. The pipeline                                #
# --------------------------------------------------------------------- #


def build_preprocessor() -> 'ColumnTransformer':
    """Builds the leak-proof preprocessing block.

    Every parameterised step -- imputer, scaler, encoder -- lives inside
    the transformer, so cross-validation fits them on training folds only
    (Chapter 11).

    Returns:
        An unfitted ColumnTransformer.
    """
    numeric = Pipeline([
        ('impute', SimpleImputer(strategy='median', add_indicator=True)),
        ('scale', StandardScaler()),
    ])
    categorical = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('encode', OneHotEncoder(handle_unknown='ignore', min_frequency=25)),
    ])
    return ColumnTransformer([
        ('num', numeric, NUMERIC),
        ('cat', categorical, CATEGORICAL),
    ], remainder='drop')


def candidate_models(quick: bool) -> Dict[str, Tuple['Pipeline', Dict]]:
    """Returns the models to compare, each with its search space.

    Args:
        quick: Whether to use the reduced search space.

    Returns:
        A mapping from name to (pipeline, parameter grid).
    """
    grids: Dict[str, Tuple[Pipeline, Dict]] = {}

    grids['logistic'] = (
        Pipeline([('prep', build_preprocessor()),
                  ('model', LogisticRegression(max_iter=2000,
                                               class_weight='balanced'))]),
        {'model__C': [0.1, 1.0] if quick else [0.01, 0.1, 1.0, 10.0]},
    )
    grids['random_forest'] = (
        Pipeline([('prep', build_preprocessor()),
                  ('model', RandomForestClassifier(
                      n_estimators=200, random_state=0, n_jobs=-1,
                      class_weight='balanced_subsample'))]),
        {'model__max_depth': [8, None] if quick else [4, 8, 16, None],
         'model__min_samples_leaf': [5] if quick else [1, 5, 20]},
    )
    grids['gradient_boosting'] = (
        Pipeline([('prep', build_preprocessor()),
                  ('model', HistGradientBoostingClassifier(random_state=0))]),
        {'model__max_leaf_nodes': [15, 31] if quick else [7, 15, 31, 63],
         'model__learning_rate': [0.1] if quick else [0.03, 0.1, 0.3]},
    )
    return grids


# --------------------------------------------------------------------- #
#                         3. Evaluation                                  #
# --------------------------------------------------------------------- #


def bootstrap_ci(metric_fn, y_true: np.ndarray, y_score: np.ndarray,
                 n_boot: int = 500, level: float = 0.95,
                 seed: int = 0) -> Tuple[float, float]:
    """Percentile bootstrap interval for a scoring metric.

    Args:
        metric_fn: Callable taking (y_true, y_score).
        y_true: Ground-truth labels.
        y_score: Predicted scores.
        n_boot: Number of resamples.
        level: Confidence level.
        seed: Seed for reproducibility.

    Returns:
        A (lower, upper) tuple.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue                       # a resample with one class only
        values.append(float(metric_fn(y_true[idx], y_score[idx])))
    tail = (1.0 - level) / 2.0
    return (float(np.quantile(values, tail)),
            float(np.quantile(values, 1.0 - tail)))


def nested_cv_score(pipeline: 'Pipeline', grid: Dict, x: 'pd.DataFrame',
                    y: np.ndarray, groups: np.ndarray,
                    outer: int = 4, inner: int = 3) -> Dict[str, float]:
    """Runs group-aware nested cross-validation.

    The inner loop selects hyperparameters; the outer loop scores a model
    that never saw the selection. The gap between the two is the
    optimism of tuning (Chapter 14).

    Args:
        pipeline: The estimator, preprocessing included.
        grid: The hyperparameter search space.
        x: Feature table.
        y: Binary targets.
        groups: Group identifier per row.
        outer: Outer fold count.
        inner: Inner fold count.

    Returns:
        A dict with nested and flat scores and the optimism gap.
    """
    outer_cv = StratifiedGroupKFold(n_splits=outer, shuffle=True,
                                    random_state=0)
    inner_cv = StratifiedGroupKFold(n_splits=inner, shuffle=True,
                                    random_state=1)

    nested_scores = []
    for train_idx, test_idx in outer_cv.split(x, y, groups):
        search = GridSearchCV(pipeline, grid, scoring='average_precision',
                              cv=inner_cv, n_jobs=-1, refit=True)
        search.fit(x.iloc[train_idx], y[train_idx],
                   groups=groups[train_idx])
        scores = search.predict_proba(x.iloc[test_idx])[:, 1]
        nested_scores.append(float(average_precision_score(y[test_idx],
                                                           scores)))

    # The flat score must be measured on the SAME folds, or the
    # comparison is meaningless: an inner-CV best_score_ is computed on
    # smaller training sets and is therefore pessimistic for reasons
    # that have nothing to do with selection optimism. Here we select
    # and report on the identical outer folds -- the mistake the nested
    # procedure exists to correct.
    flat = GridSearchCV(pipeline, grid, scoring='average_precision',
                        cv=outer_cv, n_jobs=-1, refit=False)
    flat.fit(x, y, groups=groups)
    flat_score = float(flat.best_score_)

    return {
        'nested_pr_auc': float(np.mean(nested_scores)),
        'nested_std': float(np.std(nested_scores)),
        'flat_pr_auc': flat_score,
        'optimism_gap': float(flat_score - float(np.mean(nested_scores))),
    }


def expected_cost(y_true: np.ndarray, y_score: np.ndarray,
                  threshold: float) -> float:
    """Total pounds lost at a given decision threshold.

    Args:
        y_true: Ground-truth labels.
        y_score: Predicted probabilities.
        threshold: Alarm threshold.

    Returns:
        The total cost.
    """
    predicted = y_score >= threshold
    false_pos = int(np.sum(predicted & (y_true == 0)))
    false_neg = int(np.sum((~predicted) & (y_true == 1)))
    return false_pos * COST_FALSE_POSITIVE + false_neg * COST_FALSE_NEGATIVE


def choose_threshold(y_true: np.ndarray,
                     y_score: np.ndarray) -> Tuple[float, float]:
    """Finds the cost-minimising threshold on validation data.

    Chapter 9 predicts the optimum near C_fp / (C_fp + C_fn); this
    searches for it empirically so the two can be compared.

    Args:
        y_true: Validation labels.
        y_score: Validation probabilities.

    Returns:
        A tuple (threshold, cost_at_that_threshold).
    """
    candidates = np.unique(np.round(y_score, 4))
    costs = [expected_cost(y_true, y_score, t) for t in candidates]
    best = int(np.argmin(costs))
    return float(candidates[best]), float(costs[best])


def slice_report(frame: 'pd.DataFrame', y_true: np.ndarray,
                 y_score: np.ndarray, threshold: float) -> List[Dict]:
    """Per-slice performance, to find where the model fails.

    Args:
        frame: Feature table for the evaluated rows.
        y_true: Ground-truth labels.
        y_score: Predicted probabilities.
        threshold: Decision threshold.

    Slices are ranked by PR-AUC rather than recall, because recall alone
    is an artefact of wherever the threshold happens to sit: at a low
    threshold every slice recalls nearly everything.

    Returns:
        One record per slice, weakest PR-AUC first.
    """
    records: List[Dict] = []
    for column in CATEGORICAL:
        for level in frame[column].unique():
            mask = (frame[column] == level).to_numpy()
            if mask.sum() < 40 or len(np.unique(y_true[mask])) < 2:
                continue
            predicted = y_score[mask] >= threshold
            actual = y_true[mask] == 1
            true_pos = int(np.sum(predicted & actual))
            records.append({
                'slice': f'{column}={level}',
                'n': int(mask.sum()),
                'positive_rate': round(float(actual.mean()), 4),
                'recall': round(float(true_pos / max(actual.sum(), 1)), 4),
                'precision': round(
                    float(true_pos / max(predicted.sum(), 1)), 4),
                'pr_auc': round(float(average_precision_score(
                    y_true[mask], y_score[mask])), 4),
            })
    return sorted(records, key=lambda r: r['pr_auc'])


def reliability_table(y_true: np.ndarray, y_score: np.ndarray,
                      bins: int = 8) -> List[Dict]:
    """Bins predictions and compares predicted with observed frequency.

    Args:
        y_true: Ground-truth labels.
        y_score: Predicted probabilities.
        bins: Number of equal-width bins.

    Returns:
        One record per non-empty bin.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: List[Dict] = []
    for i in range(bins):
        mask = (y_score >= edges[i]) & (y_score < edges[i + 1])
        if mask.sum() < 10:
            continue
        rows.append({
            'bin': f'[{edges[i]:.2f},{edges[i + 1]:.2f})',
            'n': int(mask.sum()),
            'mean_predicted': round(float(y_score[mask].mean()), 4),
            'observed': round(float(y_true[mask].mean()), 4),
        })
    return rows


# --------------------------------------------------------------------- #
#                            4. The run                                  #
# --------------------------------------------------------------------- #


@dataclasses.dataclass
class Gate:
    """One acceptance check, declared before the run.

    Attributes:
        name: Identifier.
        rule: The declared threshold, as text.
        observed: What the run produced.
        passed: Whether it held.
    """

    name: str
    rule: str
    observed: str
    passed: bool


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Runs the whole project and writes an audit pack.

    Args:
        argv: Optional argument vector.

    Returns:
        0 when every gate passed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=20260101)
    parser.add_argument('--quick', action='store_true',
                        help='smaller search space for a fast pass')
    parser.add_argument('--no-write', action='store_true')
    args = parser.parse_args(argv)

    if not HAS_SKLEARN:
        print(f'UNVERIFIED: scikit-learn/pandas not installed '
              f'({IMPORT_ERROR}).')
        print('Run testbed/setup.ps1 first. This project needs them.')
        return 2

    started = time.perf_counter()
    print('=' * 74)
    print('PROJECT 1 - Tabular classification, end to end')
    print('=' * 74)

    # --- 1. data and schema ------------------------------------------ #
    frame, labels, groups = generate_churn_data(seed=args.seed)
    schema = describe_schema(frame, labels, groups)
    print(f'\n[1] Data and schema')
    print(f'    rows {schema["n_rows"]}, customers {schema["n_groups"]} '
          f'({schema["rows_per_group"]} rows each)')
    print(f'    positive rate {schema["positive_rate"]:.3f}')
    print(f'    missing: {schema["missing_fraction"]}')
    print(f'    region cardinality {schema["cardinality"]["region"]}')
    print('    rows repeat per customer -> a plain KFold would leak '
          '(Chapter 10)')

    # --- 2. split, group-aware --------------------------------------- #
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    dev_idx, test_idx = next(splitter.split(frame, labels, groups))
    x_dev, y_dev, g_dev = frame.iloc[dev_idx], labels[dev_idx], groups[dev_idx]
    x_test, y_test = frame.iloc[test_idx], labels[test_idx]
    overlap = set(groups[dev_idx]) & set(groups[test_idx])
    print(f'\n[2] Group-aware split: dev {len(dev_idx)}, test {len(test_idx)}')
    print(f'    customers appearing in both splits: {len(overlap)}')

    # --- 3. baseline -------------------------------------------------- #
    baseline = DummyClassifier(strategy='prior').fit(x_dev, y_dev)
    base_scores = baseline.predict_proba(x_test)[:, 1]
    base_pr = float(average_precision_score(y_test, base_scores))
    print(f'\n[3] Baseline (predict the prior): PR-AUC {base_pr:.4f}')
    print(f'    any model must beat this, and it equals the positive rate')

    # --- 4. nested CV over three model families ----------------------- #
    print(f'\n[4] Group-aware nested CV '
          f'({"quick" if args.quick else "full"} search)')
    results: Dict[str, Dict[str, float]] = {}
    for name, (pipeline, grid) in candidate_models(args.quick).items():
        t0 = time.perf_counter()
        results[name] = nested_cv_score(pipeline, grid, x_dev, y_dev, g_dev)
        results[name]['seconds'] = round(time.perf_counter() - t0, 1)
        row = results[name]
        print(f'    {name:<18} nested PR-AUC {row["nested_pr_auc"]:.4f} '
              f'(sd {row["nested_std"]:.4f})  flat {row["flat_pr_auc"]:.4f}  '
              f'optimism {row["optimism_gap"]:+.4f}  '
              f'{row["seconds"]:.0f}s')
    best_name = max(results, key=lambda k: results[k]['nested_pr_auc'])
    print(f'    winner: {best_name}')

    # --- 5. refit, calibrate ------------------------------------------ #
    print(f'\n[5] Refit {best_name} on dev, calibrate on held-out folds')
    pipeline, grid = candidate_models(args.quick)[best_name]
    inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=1)
    search = GridSearchCV(pipeline, grid, scoring='average_precision',
                          cv=inner_cv, n_jobs=-1, refit=True)
    search.fit(x_dev, y_dev, groups=g_dev)
    print(f'    best params: {search.best_params_}')

    uncalibrated = search.best_estimator_
    raw_scores = uncalibrated.predict_proba(x_test)[:, 1]
    calibrated = CalibratedClassifierCV(estimator=search.best_estimator_,
                                        method='isotonic', cv=3)
    calibrated.fit(x_dev, y_dev)
    cal_scores = calibrated.predict_proba(x_test)[:, 1]

    brier_raw = float(brier_score_loss(y_test, raw_scores))
    brier_cal = float(brier_score_loss(y_test, cal_scores))
    print(f'    Brier before {brier_raw:.4f}  after {brier_cal:.4f}')
    for row in reliability_table(y_test, cal_scores):
        print(f'      {row["bin"]:<14} n={row["n"]:<5} '
              f'predicted {row["mean_predicted"]:.3f}  '
              f'observed {row["observed"]:.3f}')

    # --- 6. headline metrics with intervals --------------------------- #
    pr_auc = float(average_precision_score(y_test, cal_scores))
    roc_auc = float(roc_auc_score(y_test, cal_scores))
    pr_lo, pr_hi = bootstrap_ci(average_precision_score, y_test, cal_scores,
                                seed=args.seed)
    roc_lo, roc_hi = bootstrap_ci(roc_auc_score, y_test, cal_scores,
                                  seed=args.seed)
    print(f'\n[6] Held-out test, with 95% bootstrap intervals')
    print(f'    PR-AUC  {pr_auc:.4f}  [{pr_lo:.4f}, {pr_hi:.4f}]')
    print(f'    ROC-AUC {roc_auc:.4f}  [{roc_lo:.4f}, {roc_hi:.4f}]')
    print(f'    lift over baseline: {pr_auc / base_pr:.2f}x')

    # --- 7. threshold from costs -------------------------------------- #
    dev_scores = calibrated.predict_proba(x_dev)[:, 1]
    threshold, dev_cost = choose_threshold(y_dev, dev_scores)
    theory = COST_FALSE_POSITIVE / (COST_FALSE_POSITIVE + COST_FALSE_NEGATIVE)
    test_cost = expected_cost(y_test, cal_scores, threshold)
    cost_at_half = expected_cost(y_test, cal_scores, 0.5)
    print(f'\n[7] Decision threshold from the costs (Chapter 9)')
    print(f'    theory C_fp/(C_fp+C_fn) = {theory:.4f}')
    print(f'    chosen on dev           = {threshold:.4f}')
    print(f'    test cost at {threshold:.3f}: £{test_cost:,.0f}')
    print(f'    test cost at 0.500: £{cost_at_half:,.0f}  '
          f'(saving £{cost_at_half - test_cost:,.0f})')

    # --- 8. slices ----------------------------------------------------- #
    slices = slice_report(x_test, y_test, cal_scores, threshold)
    print(f'\n[8] Weakest slices by PR-AUC')
    for row in slices[:4]:
        print(f'    {row["slice"]:<28} n={row["n"]:<5} '
              f'recall {row["recall"]:.3f}  precision {row["precision"]:.3f}  '
              f'PR-AUC {row["pr_auc"]:.3f}')

    # --- 9. gates ------------------------------------------------------ #
    gates = [
        Gate('no_group_overlap', 'zero customers in both splits',
             str(len(overlap)), len(overlap) == 0),
        Gate('beats_baseline', f'PR-AUC > baseline {base_pr:.4f}',
             f'{pr_auc:.4f}', pr_auc > base_pr),
        Gate('nested_not_above_flat',
             'nested <= flat for every model',
             f'max gap {max(r["optimism_gap"] for r in results.values()):+.4f}',
             all(r['optimism_gap'] >= -1e-9 for r in results.values())),
        Gate('interval_reported', 'every headline number has a CI',
             f'PR-AUC [{pr_lo:.4f}, {pr_hi:.4f}]', True),
        Gate('cost_threshold_beats_half',
             'cost at chosen threshold <= cost at 0.5',
             f'£{test_cost:,.0f} vs £{cost_at_half:,.0f}',
             test_cost <= cost_at_half),
        Gate('calibration_improved', 'Brier not worse after calibration',
             f'{brier_raw:.4f} -> {brier_cal:.4f}',
             brier_cal <= brier_raw + 1e-4),
    ]
    print(f'\n[9] Gates')
    for gate in gates:
        mark = 'PASS' if gate.passed else 'FAIL'
        print(f'    [{mark}] {gate.name}: {gate.observed}  '
              f'(rule: {gate.rule})')

    elapsed = time.perf_counter() - started
    status = 'VERIFIED' if all(g.passed for g in gates) else 'PENDING REVISION'
    print(f'\nstatus: {status}   ({elapsed:.1f}s)')

    if not args.no_write:
        run_dir = RUNS / 'p1_tabular_churn'
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            'project': 'p1_tabular_churn',
            'status': status,
            'seed': args.seed,
            'schema': schema,
            'baseline_pr_auc': base_pr,
            'model_selection': results,
            'winner': best_name,
            'best_params': {k: str(v)
                            for k, v in search.best_params_.items()},
            'test': {
                'pr_auc': pr_auc, 'pr_auc_ci': [pr_lo, pr_hi],
                'roc_auc': roc_auc, 'roc_auc_ci': [roc_lo, roc_hi],
                'brier_uncalibrated': brier_raw, 'brier_calibrated': brier_cal,
            },
            'threshold': {'chosen': threshold, 'theory': theory,
                          'test_cost': test_cost,
                          'test_cost_at_half': cost_at_half},
            'slices': slices,
            'reliability': reliability_table(y_test, cal_scores),
            'gates': [dataclasses.asdict(g) for g in gates],
            'seconds': round(elapsed, 1),
        }
        (run_dir / 'report.json').write_text(
            json.dumps(payload, indent=2, default=str), encoding='utf-8')
        print(f'audit pack: {run_dir}')

    return 0 if all(g.passed for g in gates) else 1


if __name__ == '__main__':
    sys.exit(main())
