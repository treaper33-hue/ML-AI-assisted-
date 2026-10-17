"""Splitters, and the assertions that make leaks impossible.

Leakage arbitration (ledger GAP-08): [TSP p.158-p.164] is canonical for this
program, including for non-temporal work. [SKL ch.3 "Leakage-Safe
Evaluation"] supplies the scikit-learn mechanics.

Every splitter returns `Fold` objects that carry a human-readable boundary
description, so a reviewer can check the split without reading the code.
"""

import dataclasses
from typing import Iterable, List, Tuple

import numpy as np


@dataclasses.dataclass(frozen=True)
class Fold:
    """One train/test partition.

    Attributes:
        name: Identifier such as 'fold_0'.
        train_idx: Indices used for fitting.
        test_idx: Indices used for scoring.
        boundary: Human-readable description for the run report.
    """

    name: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    boundary: str

    def describe(self) -> str:
        """Returns a one-line summary for the report."""
        return (
            f'{self.name}: train n={self.train_idx.size} '
            f'test n={self.test_idx.size} | {self.boundary}'
        )


def random_split(
    n_samples: int,
    test_size: float = 0.2,
    seed: int = 20260101,
) -> Fold:
    """Shuffled split for i.i.d. tabular data [SKL ch.3 "Stratified Splits"].

    Args:
        n_samples: Number of rows.
        test_size: Fraction held out.
        seed: Seed for the shuffle.

    Returns:
        A single `Fold`.

    Raises:
        ValueError: If `test_size` is not strictly between 0 and 1.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError('test_size must be in (0, 1)')
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_samples)
    n_test = max(1, int(round(n_samples * test_size)))
    test_idx = np.sort(order[:n_test])
    train_idx = np.sort(order[n_test:])
    return Fold(
        name='random',
        train_idx=train_idx,
        test_idx=test_idx,
        boundary=f'shuffled, test_size={test_size}, seed={seed}',
    )


def fixed_origin_split(n_samples: int, horizon: int) -> Fold:
    """Fixed-origin temporal split [TSP p.126].

    Args:
        n_samples: Length of the series.
        horizon: Number of final steps held out.

    Returns:
        A single `Fold` whose train set strictly precedes its test set.

    Raises:
        ValueError: If the horizon leaves no training data.
    """
    if horizon <= 0 or horizon >= n_samples:
        raise ValueError('horizon must be in (0, n_samples)')
    cut = n_samples - horizon
    return Fold(
        name='fixed_origin',
        train_idx=np.arange(cut),
        test_idx=np.arange(cut, n_samples),
        boundary=f'train [0,{cut}) < test [{cut},{n_samples})',
    )


def expanding_window_cv(
    n_samples: int,
    n_folds: int,
    horizon: int,
    min_train: int = 0,
) -> List[Fold]:
    """Expanding-window cross-validation [TSP p.132].

    The training window grows with each fold; the origin moves forward.

    Args:
        n_samples: Length of the series.
        n_folds: Number of folds.
        horizon: Test length per fold.
        min_train: Minimum initial training length; defaults to one horizon.

    Returns:
        A list of folds in chronological order.

    Raises:
        ValueError: If the series is too short for the requested design.
    """
    start = min_train or horizon
    needed = start + n_folds * horizon
    if needed > n_samples:
        raise ValueError(
            f'need {needed} points for {n_folds} folds at horizon {horizon}, '
            f'have {n_samples}'
        )
    folds: List[Fold] = []
    for i in range(n_folds):
        cut = start + i * horizon
        folds.append(
            Fold(
                name=f'expanding_{i}',
                train_idx=np.arange(cut),
                test_idx=np.arange(cut, cut + horizon),
                boundary=f'train [0,{cut}) < test [{cut},{cut + horizon})',
            )
        )
    return folds


def rolling_window_cv(
    n_samples: int,
    n_folds: int,
    horizon: int,
    window: int,
) -> List[Fold]:
    """Rolling-window cross-validation [TSP p.134].

    The training window has a fixed length and slides forward, which keeps
    the amount of history constant across folds.

    Args:
        n_samples: Length of the series.
        n_folds: Number of folds.
        horizon: Test length per fold.
        window: Fixed training-window length.

    Returns:
        A list of folds in chronological order.

    Raises:
        ValueError: If the series is too short for the requested design.
    """
    needed = window + n_folds * horizon
    if needed > n_samples:
        raise ValueError(
            f'need {needed} points for {n_folds} folds at horizon {horizon} '
            f'with window {window}, have {n_samples}'
        )
    folds: List[Fold] = []
    for i in range(n_folds):
        end = window + i * horizon
        folds.append(
            Fold(
                name=f'rolling_{i}',
                train_idx=np.arange(end - window, end),
                test_idx=np.arange(end, end + horizon),
                boundary=(
                    f'train [{end - window},{end}) < '
                    f'test [{end},{end + horizon})'
                ),
            )
        )
    return folds


def assert_no_lookforward(folds: Iterable[Fold]) -> None:
    """Fails loudly if any training index reaches past its test window.

    This is the mechanical half of the lookforward-bias check described at
    [TSP p.161]; the other half is making sure transformations are fitted
    inside the fold [TSP p.159].

    Args:
        folds: Folds to check.

    Raises:
        AssertionError: If a fold leaks the future into training.
    """
    for fold in folds:
        if fold.train_idx.size == 0 or fold.test_idx.size == 0:
            raise AssertionError(f'{fold.name}: empty train or test split')
        if fold.train_idx.max() >= fold.test_idx.min():
            raise AssertionError(
                f'{fold.name}: lookforward leak -- max train index '
                f'{fold.train_idx.max()} >= min test index '
                f'{fold.test_idx.min()} ([TSP p.161])'
            )


def assert_disjoint(folds: Iterable[Fold]) -> None:
    """Fails if any fold shares rows between train and test.

    Args:
        folds: Folds to check.

    Raises:
        AssertionError: If train and test overlap.
    """
    for fold in folds:
        overlap = np.intersect1d(fold.train_idx, fold.test_idx)
        if overlap.size:
            raise AssertionError(
                f'{fold.name}: {overlap.size} row(s) in both train and test'
            )


def summarise(folds: Iterable[Fold]) -> Tuple[str, ...]:
    """Returns one description line per fold, for the run report."""
    return tuple(fold.describe() for fold in folds)
