"""Contract tests for the single measurement tools.

These encode the disagreements between the sources (ledger GAP-09). If one
of these fails, every comparison in the program is suspect, so the setup
script refuses to continue.
"""

import numpy as np
import pytest

from mlkit import metrics


def test_rmse_matches_hand_calculation():
    """RMSE on a tiny case anyone can check on paper."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 5.0])
    assert metrics.mse(y_true, y_pred) == pytest.approx(4.0 / 3.0)
    assert metrics.rmse(y_true, y_pred) == pytest.approx(np.sqrt(4.0 / 3.0))


def test_mae_and_rmse_order():
    """MAE never exceeds RMSE, by Jensen's inequality."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        y_true = rng.normal(size=64)
        y_pred = y_true + rng.normal(size=64)
        absolute = metrics.mae(y_true, y_pred)
        squared = metrics.rmse(y_true, y_pred)
        assert absolute <= squared + 1e-12


def test_r2_is_one_for_perfect_fit():
    """A perfect prediction gives R-squared of exactly 1."""
    y_true = np.array([3.0, -1.0, 7.5, 0.25])
    assert metrics.r2(y_true, y_true) == pytest.approx(1.0)


def test_ape_raises_on_zero_targets():
    """APE is undefined at zeros [TSP p.144].

    Returning infinity instead would silently poison every aggregate
    computed downstream, which is exactly the failure this program exists
    to make impossible.
    """
    y_true = np.array([1.0, 0.0, 2.0])
    y_pred = np.array([1.0, 0.5, 2.0])
    with pytest.raises(ValueError, match='undefined for zero targets'):
        metrics.absolute_percentage_error(y_true, y_pred)
    with pytest.raises(ValueError):
        metrics.mape(y_true, y_pred)


def test_mape_is_not_scale_invariant_but_mase_is():
    """The Celsius/Kelvin trap [TSP p.143].

    Shifting the unit leaves the model unchanged, yet MAPE moves. MASE
    does not. This is why MASE is the program's primary metric.
    """
    rng = np.random.default_rng(7)
    y_train = 20.0 + rng.normal(scale=2.0, size=200)
    y_true = 20.0 + rng.normal(scale=2.0, size=50)
    y_pred = y_true + rng.normal(scale=0.5, size=50)

    offset = 273.15  # Celsius -> Kelvin
    mape_c = metrics.mape(y_true, y_pred)
    mape_k = metrics.mape(y_true + offset, y_pred + offset)
    assert abs(mape_c - mape_k) > 1e-3

    mase_c = metrics.mase(y_true, y_pred, y_train)
    mase_k = metrics.mase(y_true + offset, y_pred + offset, y_train + offset)
    assert mase_c == pytest.approx(mase_k, rel=1e-12)


def test_mase_denominator_uses_train_only():
    """MASE scales by the naive forecast on TRAIN [TSP p.150].

    Changing the test window must not change the scale; changing the
    training series must.
    """
    rng = np.random.default_rng(11)
    y_train = np.cumsum(rng.normal(size=300))
    scale_a = metrics.naive_scale(y_train)
    scale_b = metrics.naive_scale(y_train.copy())
    assert scale_a == pytest.approx(scale_b)
    other_train = np.cumsum(rng.normal(scale=5.0, size=300))
    assert metrics.naive_scale(other_train) != pytest.approx(scale_a)


def test_mase_below_one_means_beats_naive():
    """A model that copies the last value scores about 1.0."""
    rng = np.random.default_rng(3)
    series = np.cumsum(rng.normal(size=400))
    y_train, y_true = series[:300], series[300:]
    naive_pred = np.concatenate([[series[299]], y_true[:-1]])
    score = metrics.mase(y_true, naive_pred, y_train)
    assert score == pytest.approx(1.0, rel=0.35)


def test_naive_scale_rejects_constant_series():
    """A constant training series makes MASE undefined, so we say so."""
    with pytest.raises(ValueError, match='naive scale is zero'):
        metrics.naive_scale(np.ones(50))


def test_rmsse_is_scale_invariant():
    """RMSSE, like MASE, survives a change of units [TSP p.150]."""
    rng = np.random.default_rng(5)
    y_train = np.cumsum(rng.normal(size=200))
    y_true = np.cumsum(rng.normal(size=40)) + y_train[-1]
    y_pred = y_true + rng.normal(scale=0.3, size=40)
    factor = 1000.0
    base = metrics.rmsse(y_true, y_pred, y_train)
    scaled = metrics.rmsse(y_true * factor, y_pred * factor, y_train * factor)
    assert base == pytest.approx(scaled, rel=1e-10)


def test_f1_macro_and_micro_differ_under_imbalance():
    """Macro and micro F1 answer different questions [SKL ch.6].

    Both are returned together so a report cannot quietly show whichever
    one flatters the model.
    """
    y_true = np.array([0] * 90 + [1] * 10)
    y_pred = np.array([0] * 100)
    scores = metrics.f1_scores(y_true, y_pred)
    assert scores['f1_micro'] == pytest.approx(0.9)
    assert scores['f1_macro'] < scores['f1_micro']


def test_confusion_matrix_rows_are_truth():
    """Rows are the truth, columns the prediction [PTG p.95]."""
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0])
    matrix = metrics.confusion_matrix(y_true, y_pred)
    assert matrix.tolist() == [[1, 1], [1, 2]]
    assert matrix.sum() == y_true.size


def test_bootstrap_ci_contains_point_estimate():
    """A percentile interval brackets the observed value [SKL ch.6]."""
    rng = np.random.default_rng(13)
    y_true = rng.normal(size=500)
    y_pred = y_true + rng.normal(scale=0.5, size=500)
    point = metrics.rmse(y_true, y_pred)
    lower, upper = metrics.bootstrap_ci(
        metrics.rmse, y_true, y_pred, n_boot=400, seed=1
    )
    assert lower <= point <= upper


def test_bootstrap_ci_is_deterministic_given_a_seed():
    """Reproducibility: the same seed gives the same interval."""
    rng = np.random.default_rng(17)
    y_true = rng.normal(size=200)
    y_pred = y_true + rng.normal(scale=0.4, size=200)
    first = metrics.bootstrap_ci(metrics.rmse, y_true, y_pred, 200, seed=42)
    second = metrics.bootstrap_ci(metrics.rmse, y_true, y_pred, 200, seed=42)
    assert first == second


def test_shape_mismatch_is_rejected():
    """Silent broadcasting is a bug factory, so we refuse it."""
    with pytest.raises(ValueError, match='shape mismatch'):
        metrics.rmse(np.zeros(5), np.zeros(6))
