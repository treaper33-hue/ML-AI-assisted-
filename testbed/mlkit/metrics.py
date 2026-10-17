"""The single measurement tools.

Every number reported anywhere in the program comes from this module. Three
sources in the stack define overlapping metrics and disagree at the edges,
so the disagreements are resolved here, once, with the resolution recorded
next to the code (ledger GAP-09).

Resolutions:
    * APE is undefined where the truth is zero [TSP p.144]: we raise rather
      than silently returning infinity.
    * MAPE is not scale invariant -- it changes under a unit change
      [TSP p.143]. MASE is; `test_metrics.py` proves both.
    * MASE and RMSSE scale by the naive forecast computed on TRAIN ONLY
      [TSP p.150]. Using test data in the denominator is a leak.
    * Macro and micro F1 answer different questions [SKL ch.6]; both are
      returned, never one silently.
"""

from typing import Callable, Dict, Tuple

import numpy as np


def _as_1d(array: np.ndarray, name: str) -> np.ndarray:
    """Validates and flattens an input array.

    Args:
        array: Candidate array.
        name: Name used in error messages.

    Returns:
        A 1-D float array.

    Raises:
        ValueError: If the array is empty or not 1-D after squeezing.
    """
    values = np.asarray(array, dtype=np.float64).squeeze()
    if values.ndim == 0:
        values = values.reshape(1)
    if values.ndim != 1:
        raise ValueError(f'{name} must be 1-D, got shape {values.shape}')
    if values.size == 0:
        raise ValueError(f'{name} must not be empty')
    return values


def _check_pair(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Validates a truth/prediction pair.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        The validated pair as 1-D float arrays.

    Raises:
        ValueError: If the shapes do not match.
    """
    truth = _as_1d(y_true, 'y_true')
    pred = _as_1d(y_pred, 'y_pred')
    if truth.shape != pred.shape:
        raise ValueError(
            f'shape mismatch: y_true {truth.shape} vs y_pred {pred.shape}'
        )
    return truth, pred


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error [TSP p.138]."""
    truth, pred = _check_pair(y_true, y_pred)
    return float(np.mean(np.abs(truth - pred)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error [TSP p.144]."""
    truth, pred = _check_pair(y_true, y_pred)
    return float(np.mean((truth - pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error [TSP p.144]."""
    return float(np.sqrt(mse(y_true, y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination [SKL ch.6 "Regression"].

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        R-squared. Returns 0.0 when the truth has zero variance, since the
        ratio is undefined there.
    """
    truth, pred = _check_pair(y_true, y_pred)
    total = float(np.sum((truth - np.mean(truth)) ** 2))
    if total == 0.0:
        return 0.0
    residual = float(np.sum((truth - pred) ** 2))
    return 1.0 - residual / total


def absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Element-wise absolute percentage error [TSP p.139].

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        The element-wise APE, as a fraction (not multiplied by 100).

    Raises:
        ValueError: If any ground-truth value is zero. APE is undefined
            there [TSP p.144]; returning infinity would quietly poison every
            aggregate computed downstream.
    """
    truth, pred = _check_pair(y_true, y_pred)
    if np.any(truth == 0.0):
        n_zeros = int(np.sum(truth == 0.0))
        raise ValueError(
            f'APE is undefined for zero targets ([TSP p.144]); '
            f'{n_zeros} zero value(s) in y_true. Use MASE or RMSSE instead.'
        )
    return np.asarray(np.abs((truth - pred) / truth), dtype=np.float64)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error [TSP p.139].

    Not scale invariant: see the Celsius/Kelvin table at [TSP p.143].
    """
    return float(np.mean(absolute_percentage_error(y_true, y_pred)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE [TSP p.147 "Variations"].

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        sMAPE as a fraction. Points where both truth and prediction are
        zero contribute zero rather than a division by zero.
    """
    truth, pred = _check_pair(y_true, y_pred)
    denominator = (np.abs(truth) + np.abs(pred)) / 2.0
    ratio = np.zeros_like(denominator)
    nonzero = denominator != 0.0
    numerator = np.abs(truth[nonzero] - pred[nonzero])
    ratio[nonzero] = numerator / denominator[nonzero]
    return float(np.mean(ratio))


def naive_scale(y_train: np.ndarray, seasonality: int = 1) -> float:
    """Mean absolute error of the seasonal naive forecast on TRAIN.

    This is the denominator of MASE [TSP p.150]. It is computed on the
    training series only; computing it on test data is a leak.

    Args:
        y_train: Training portion of the series, in time order.
        seasonality: Seasonal period; 1 gives the random-walk baseline.

    Returns:
        The scaling constant.

    Raises:
        ValueError: If the series is too short, or the scale is zero.
    """
    train = _as_1d(y_train, 'y_train')
    if seasonality < 1:
        raise ValueError('seasonality must be >= 1')
    if train.size <= seasonality:
        raise ValueError(
            f'need more than {seasonality} training points, got {train.size}'
        )
    scale = float(np.mean(np.abs(train[seasonality:] - train[:-seasonality])))
    if scale == 0.0:
        raise ValueError(
            'naive scale is zero (constant training series); MASE undefined'
        )
    return scale


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 1,
) -> float:
    """Mean absolute scaled error [TSP p.150].

    Scale invariant, defined at zeros, and comparable across series --
    which is why it is this program's primary forecasting metric.

    Args:
        y_true: Ground-truth values over the evaluation window.
        y_pred: Predicted values over the evaluation window.
        y_train: Training series used for the naive scale.
        seasonality: Seasonal period for the naive baseline.

    Returns:
        MASE. Below 1.0 means the model beats the naive forecast.
    """
    return mae(y_true, y_pred) / naive_scale(y_train, seasonality)


def rmsse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 1,
) -> float:
    """Root mean squared scaled error [TSP p.150].

    Args:
        y_true: Ground-truth values over the evaluation window.
        y_pred: Predicted values over the evaluation window.
        y_train: Training series used for the naive scale.
        seasonality: Seasonal period for the naive baseline.

    Returns:
        RMSSE, the squared-error analogue of MASE.

    Raises:
        ValueError: If the squared naive scale is zero.
    """
    train = _as_1d(y_train, 'y_train')
    if train.size <= seasonality:
        raise ValueError('training series too short for the chosen seasonality')
    diffs = train[seasonality:] - train[:-seasonality]
    scale = float(np.mean(diffs**2))
    if scale == 0.0:
        raise ValueError('squared naive scale is zero; RMSSE undefined')
    return float(np.sqrt(mse(y_true, y_pred) / scale))


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int = 0,
) -> np.ndarray:
    """Confusion matrix with true labels on rows [PTG p.95].

    Args:
        y_true: Integer ground-truth labels.
        y_pred: Integer predicted labels.
        n_classes: Number of classes; inferred when 0.

    Returns:
        An (n_classes, n_classes) integer array.
    """
    truth = _as_1d(y_true, 'y_true').astype(np.int64)
    pred = _as_1d(y_pred, 'y_pred').astype(np.int64)
    size = n_classes or int(max(truth.max(), pred.max()) + 1)
    matrix = np.zeros((size, size), dtype=np.int64)
    # Shapes were validated above, so strict pairing is safe and
    # catches any future change that breaks that assumption.
    for actual, predicted in zip(truth, pred, strict=True):
        matrix[actual, predicted] += 1
    return matrix


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of correct predictions.

    Reported only alongside a class-balance figure: on imbalanced data it
    is the trap described at [SKL ch.1 "What Can Go Wrong"].
    """
    truth, pred = _check_pair(y_true, y_pred)
    return float(np.mean(truth == pred))


def f1_scores(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Macro and micro F1, always returned together [SKL ch.6].

    Args:
        y_true: Integer ground-truth labels.
        y_pred: Integer predicted labels.

    Returns:
        A dictionary with keys 'f1_macro' and 'f1_micro'.
    """
    matrix = confusion_matrix(y_true, y_pred)
    true_positive = np.diag(matrix).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    actual = matrix.sum(axis=1).astype(np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.where(predicted > 0, true_positive / predicted, 0.0)
        recall = np.where(actual > 0, true_positive / actual, 0.0)
        denominator = precision + recall
        per_class = np.where(
            denominator > 0, 2.0 * precision * recall / denominator, 0.0
        )
    micro = float(true_positive.sum() / matrix.sum()) if matrix.sum() else 0.0
    return {'f1_macro': float(np.mean(per_class)), 'f1_micro': micro}


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 1000,
    level: float = 0.95,
    seed: int = 20260101,
) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval [SKL ch.6 "Uncertainty"].

    Every headline number in this program carries one of these.

    Args:
        metric_fn: Metric taking (y_true, y_pred) and returning a float.
        y_true: Ground-truth values.
        y_pred: Predicted values.
        n_boot: Number of bootstrap resamples.
        level: Confidence level, e.g. 0.95.
        seed: Seed for the resampling generator.

    Returns:
        A (lower, upper) tuple.
    """
    truth, pred = _check_pair(y_true, y_pred)
    rng = np.random.default_rng(seed)
    n = truth.size
    samples = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples[i] = metric_fn(truth[idx], pred[idx])
    tail = (1.0 - level) / 2.0
    lower = float(np.quantile(samples, tail))
    upper = float(np.quantile(samples, 1.0 - tail))
    return lower, upper


REGRESSION_METRICS: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    'mae': mae,
    'mse': mse,
    'rmse': rmse,
    'r2': r2,
    'smape': smape,
}
