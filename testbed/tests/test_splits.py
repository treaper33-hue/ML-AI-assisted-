"""Contract tests for the splitters.

Anchors: fixed origin [TSP p.126], expanding windows [TSP p.132], rolling
windows [TSP p.134], lookforward bias [TSP p.161].
"""

import numpy as np
import pytest

from mlkit import splits


def test_random_split_is_disjoint_and_complete():
    """Every row lands in exactly one side of the split."""
    fold = splits.random_split(n_samples=100, test_size=0.2, seed=1)
    splits.assert_disjoint([fold])
    assert fold.test_idx.size == 20
    union = np.union1d(fold.train_idx, fold.test_idx)
    assert union.size == 100


def test_random_split_is_reproducible():
    """The same seed gives the same split, or no run is replayable."""
    first = splits.random_split(100, 0.25, seed=7)
    second = splits.random_split(100, 0.25, seed=7)
    assert np.array_equal(first.test_idx, second.test_idx)


def test_fixed_origin_puts_all_training_before_all_testing():
    """The defining property of a fixed-origin split [TSP p.126]."""
    fold = splits.fixed_origin_split(n_samples=100, horizon=10)
    splits.assert_no_lookforward([fold])
    assert fold.train_idx.max() == 89
    assert fold.test_idx.min() == 90


def test_expanding_windows_grow_and_never_look_forward():
    """Training history grows while the origin moves [TSP p.132]."""
    folds = splits.expanding_window_cv(
        n_samples=120, n_folds=4, horizon=10, min_train=50
    )
    splits.assert_no_lookforward(folds)
    splits.assert_disjoint(folds)
    sizes = [fold.train_idx.size for fold in folds]
    assert sizes == sorted(sizes)
    assert sizes[0] == 50


def test_rolling_windows_keep_a_constant_history():
    """Training length stays fixed as the window slides [TSP p.134]."""
    folds = splits.rolling_window_cv(
        n_samples=120, n_folds=4, horizon=10, window=50
    )
    splits.assert_no_lookforward(folds)
    splits.assert_disjoint(folds)
    assert {fold.train_idx.size for fold in folds} == {50}


def test_lookforward_leak_is_detected():
    """The assertion must actually fire on a bad fold [TSP p.161]."""
    bad = splits.Fold(
        name='bad',
        train_idx=np.arange(0, 60),
        test_idx=np.arange(50, 70),
        boundary='deliberately overlapping',
    )
    with pytest.raises(AssertionError, match='lookforward leak'):
        splits.assert_no_lookforward([bad])


def test_overlapping_rows_are_detected():
    """Train and test must never share a row."""
    bad = splits.Fold(
        name='bad',
        train_idx=np.array([1, 2, 3]),
        test_idx=np.array([3, 4]),
        boundary='deliberately overlapping',
    )
    with pytest.raises(AssertionError, match='both train and test'):
        splits.assert_disjoint([bad])


def test_designs_that_do_not_fit_are_rejected_loudly():
    """Asking for more folds than the series supports must fail fast."""
    with pytest.raises(ValueError, match='need'):
        splits.expanding_window_cv(n_samples=40, n_folds=10, horizon=10)
    with pytest.raises(ValueError, match='need'):
        splits.rolling_window_cv(n_samples=40, n_folds=5, horizon=10, window=30)


def test_fold_descriptions_are_human_readable():
    """A reviewer must be able to check the split without the code."""
    folds = splits.expanding_window_cv(100, 3, 10, min_train=40)
    lines = splits.summarise(folds)
    assert len(lines) == 3
    assert 'train [0,40) < test [40,50)' in lines[0]
