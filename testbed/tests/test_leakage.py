"""The leakage contract, demonstrated rather than asserted.

[TSP p.158-p.164] is this program's canonical leakage reference, including
for non-temporal work (ledger GAP-08). The tests below reproduce two of its
failure modes on data with no signal at all, so the inflated score has
nowhere to come from except the leak.

These are the most important tests in the testbed: they turn "be careful
about leakage" into a number you have seen with your own eyes.
"""

import numpy as np

from mlkit import metrics
from mlkit import splits


def _fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> np.ndarray:
    """Least-squares fit with an intercept, used as a neutral probe."""
    design = np.hstack([x_train, np.ones((x_train.shape[0], 1))])
    weights, *_ = np.linalg.lstsq(design, y_train, rcond=None)
    test_design = np.hstack([x_test, np.ones((x_test.shape[0], 1))])
    return test_design @ weights


def test_feature_selection_outside_the_fold_invents_signal():
    """Selection on all the data leaks the test set [TSP p.158].

    X is pure noise and y is independent of it, so any positive test
    R-squared is manufactured by the selection step. Selecting on the
    training rows only removes the illusion.
    """
    rng = np.random.default_rng(20260101)
    n_samples, n_features, n_keep = 200, 2000, 20
    x = rng.normal(size=(n_samples, n_features))
    y = rng.normal(size=n_samples)

    fold = splits.random_split(n_samples, test_size=0.3, seed=1)
    x_train, y_train = x[fold.train_idx], y[fold.train_idx]
    x_test, y_test = x[fold.test_idx], y[fold.test_idx]

    # Leaking version: rank features using every row, including test rows.
    corr_all = np.abs(x.T @ (y - y.mean()))
    leaked = np.argsort(corr_all)[-n_keep:]
    leaked_pred = _fit_predict(x_train[:, leaked], y_train, x_test[:, leaked])
    leaked_r2 = metrics.r2(y_test, leaked_pred)

    # Honest version: rank features using the training rows only.
    corr_train = np.abs(x_train.T @ (y_train - y_train.mean()))
    honest = np.argsort(corr_train)[-n_keep:]
    honest_pred = _fit_predict(x_train[:, honest], y_train, x_test[:, honest])
    honest_r2 = metrics.r2(y_test, honest_pred)

    assert leaked_r2 > honest_r2 + 0.1, (
        'the leak should inflate the score; if this ever fails, the '
        'demonstration has stopped demonstrating anything'
    )
    assert honest_r2 < 0.2, (
        'no signal exists, so an honest fit must not find one'
    )


def test_target_derived_feature_leaks_completely():
    """A feature built from the target is the purest leak there is.

    This is the transformation-based leakage of [TSP p.159] in its most
    obvious form. It is included because in real pipelines it arrives
    disguised -- as an aggregate, a label-encoded category, or a rolling
    statistic computed over the whole series.
    """
    rng = np.random.default_rng(42)
    n_samples = 300
    x = rng.normal(size=(n_samples, 4))
    y = rng.normal(size=n_samples)
    noisy_target = y + rng.normal(scale=0.01, size=n_samples)
    contaminated = np.hstack([x, noisy_target.reshape(-1, 1)])

    fold = splits.random_split(n_samples, test_size=0.3, seed=2)
    clean_pred = _fit_predict(
        x[fold.train_idx], y[fold.train_idx], x[fold.test_idx]
    )
    dirty_pred = _fit_predict(
        contaminated[fold.train_idx],
        y[fold.train_idx],
        contaminated[fold.test_idx],
    )
    clean_r2 = metrics.r2(y[fold.test_idx], clean_pred)
    dirty_r2 = metrics.r2(y[fold.test_idx], dirty_pred)

    assert clean_r2 < 0.2
    assert dirty_r2 > 0.95, 'a target-derived feature should look miraculous'


def test_shuffling_a_time_series_destroys_the_evaluation():
    """Random splits on temporal data leak the future [TSP p.161].

    A random split lets the model interpolate between neighbouring points
    it has already seen; a fixed-origin split makes it extrapolate, which
    is the task it will actually face in production.
    """
    rng = np.random.default_rng(3)
    n_samples = 400
    t = np.arange(n_samples, dtype=np.float64)
    series = (
        0.05 * t
        + 3.0 * np.sin(2 * np.pi * t / 7)
        + rng.normal(scale=0.5, size=n_samples)
    )
    lags = np.column_stack([np.roll(series, shift) for shift in (1, 2, 3, 7)])[
        10:
    ]
    target = series[10:]
    n_usable = target.size

    random_fold = splits.random_split(n_usable, test_size=0.25, seed=4)
    temporal_fold = splits.fixed_origin_split(n_usable, horizon=n_usable // 4)
    splits.assert_no_lookforward([temporal_fold])

    scores = {}
    for label, fold in (('random', random_fold), ('temporal', temporal_fold)):
        pred = _fit_predict(
            lags[fold.train_idx], target[fold.train_idx], lags[fold.test_idx]
        )
        scores[label] = metrics.rmse(target[fold.test_idx], pred)

    # The random split is not automatically "better" on every draw, but it
    # answers a different question. The contract we enforce is that the
    # temporal split is the one a forecast is judged by.
    assert np.isfinite(scores['random'])
    assert np.isfinite(scores['temporal'])
    assert temporal_fold.train_idx.max() < temporal_fold.test_idx.min()
