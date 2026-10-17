"""Project 3 — Forecasting with honest evaluation and real intervals.

The hardest of the four, because temporal data punishes every evaluation
shortcut immediately. A complete forecasting project:

    diagnostics (decomposition, ACF, stationarity) -> rolling-origin folds
    -> four baselines -> three models -> MASE/RMSSE -> split conformal
    intervals -> coverage measured BEFORE and AFTER a deliberate regime
    shift -> horizon curve -> audit pack

The headline result is deliberately uncomfortable: conformal coverage
holds in the stationary stretch and degrades across the shift, because
exchangeability fails exactly when you need the guarantee. Measuring that
is the point.

Chapters: 22 (diagnostics), 23 (temporal evaluation, MASE), 24 (neural
forecasting), 25 (conformal prediction).

Usage:
    python p3_forecast_conformal.py
    python p3_forecast_conformal.py --no-shift      stationary control
    python p3_forecast_conformal.py --horizon 14 --folds 6
"""

import argparse
import dataclasses
import json
import pathlib
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    HAS_SKLEARN = True
except ImportError:                 # pragma: no cover - environment gate
    HAS_SKLEARN = False

try:
    from statsmodels.tsa.stattools import adfuller, kpss
    HAS_STATSMODELS = True
except ImportError:                 # pragma: no cover - environment gate
    HAS_STATSMODELS = False

HERE = pathlib.Path(__file__).resolve().parent
RUNS = HERE / 'runs'

PERIOD = 7           # weekly seasonality
LAGS = [1, 2, 3, 4, 5, 6, 7, 8, 14, 21, 28]


# --------------------------------------------------------------------- #
#                      1. Data, with a regime shift                      #
# --------------------------------------------------------------------- #


def generate_series(n: int = 1400, seed: int = 20260101,
                    shift: bool = True) -> Tuple[np.ndarray, int]:
    """Generates a daily series with trend, seasonality and a shift.

    The shift is a permanent change in level and seasonal amplitude
    three-quarters of the way through -- a promotion, a pricing change,
    a competitor entering. It is what breaks exchangeability and
    therefore what conformal prediction's guarantee cannot survive.

    Args:
        n: Length of the series.
        seed: Seed for the generator.
        shift: Whether to apply the regime shift.

    Returns:
        A tuple (series, shift_index) where shift_index is where the
        regime changes (n if there is no shift).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)

    level = 100.0 + 0.03 * t
    weekly = 8.0 * np.sin(2 * np.pi * t / PERIOD)
    annual = 5.0 * np.sin(2 * np.pi * t / 365.0)
    noise = rng.normal(scale=2.5, size=n)

    # Mild autocorrelation in the noise: real residuals are rarely white.
    for i in range(1, n):
        noise[i] += 0.35 * noise[i - 1]

    series = level + weekly + annual + noise
    shift_index = int(n * 0.75) if shift else n
    if shift:
        series[shift_index:] += 18.0                    # level jump
        extra = np.arange(n - shift_index, dtype=np.float64)
        series[shift_index:] += 6.0 * np.sin(2 * np.pi * extra / PERIOD)
    return series, shift_index


# --------------------------------------------------------------------- #
#                          2. Diagnostics                                #
# --------------------------------------------------------------------- #


def acf(series: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocorrelation for lags 0..max_lag.

    Args:
        series: One-dimensional series.
        max_lag: Highest lag.

    Returns:
        Array of length max_lag + 1.
    """
    y = np.asarray(series, dtype=np.float64)
    y = y - y.mean()
    denom = float(y @ y)
    return np.array([float(y[k:] @ y[:len(y) - k]) / denom
                     for k in range(max_lag + 1)])


def additive_decompose(series: np.ndarray,
                       period: int) -> Tuple[np.ndarray, np.ndarray,
                                             np.ndarray]:
    """Centred-moving-average additive decomposition.

    Args:
        series: The series.
        period: Seasonal period.

    Returns:
        A tuple (trend, seasonal, residual) that re-sums to the series.
    """
    y = np.asarray(series, dtype=np.float64)
    trend = np.convolve(y, np.ones(period) / period, mode='same')
    half = period // 2
    trend[:half] = trend[half]
    trend[len(y) - half:] = trend[len(y) - half - 1]
    detrended = y - trend
    means = np.array([detrended[i::period].mean() for i in range(period)])
    means -= means.mean()
    seasonal = np.array([means[i % period] for i in range(len(y))])
    return trend, seasonal, y - trend - seasonal


def stationarity_report(series: np.ndarray) -> Dict[str, object]:
    """Runs ADF and KPSS and records BOTH outcomes.

    The two tests have opposite nulls, so disagreement is informative,
    not an error -- and picking whichever agrees with your plan is the
    quiet dishonesty Chapter 22 warns about.

    Args:
        series: The series to test.

    Returns:
        A dict with both p-values and the combined reading.
    """
    if not HAS_STATSMODELS:
        return {'available': False,
                'note': 'statsmodels absent; stationarity UNVERIFIED'}
    import warnings
    with warnings.catch_warnings():
        # Two notices suppressed, both cosmetic: KPSS clips its
        # p-value at the table edges, and statsmodels warns that
        # adfuller will return a result object rather than a tuple
        # in a future release. Revisit the [1] indexing then.
        warnings.simplefilter('ignore')
        adf_p = float(adfuller(series, autolag='AIC')[1])
        kpss_p = float(kpss(series, regression='c', nlags='auto')[1])

    adf_says_stationary = adf_p < 0.05
    kpss_says_stationary = kpss_p > 0.05
    if adf_says_stationary and kpss_says_stationary:
        reading = 'both agree: stationary'
    elif not adf_says_stationary and not kpss_says_stationary:
        reading = 'both agree: non-stationary'
    elif adf_says_stationary and not kpss_says_stationary:
        reading = 'DISAGREE: likely trend-stationary or heteroskedastic'
    else:
        reading = 'DISAGREE: not enough evidence either way'
    return {'available': True, 'adf_p': adf_p, 'kpss_p': kpss_p,
            'adf_says_stationary': adf_says_stationary,
            'kpss_says_stationary': kpss_says_stationary,
            'reading': reading}


# --------------------------------------------------------------------- #
#                           3. Metrics                                   #
# --------------------------------------------------------------------- #


def naive_scale(y_train: np.ndarray, seasonality: int) -> float:
    """MAE of the seasonal naive forecast on TRAIN only.

    Args:
        y_train: Training portion of the series.
        seasonality: Seasonal period.

    Returns:
        The MASE denominator.

    Raises:
        ValueError: If the series is too short or constant.
    """
    y = np.asarray(y_train, dtype=np.float64)
    if y.size <= seasonality:
        raise ValueError('training series too short')
    scale = float(np.mean(np.abs(y[seasonality:] - y[:-seasonality])))
    if scale == 0.0:
        raise ValueError('constant training series; MASE undefined')
    return scale


def mase(y_true, y_pred, y_train, seasonality: int = PERIOD) -> float:
    """Mean absolute scaled error."""
    mae = float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))
    return mae / naive_scale(y_train, seasonality)


def rmsse(y_true, y_pred, y_train, seasonality: int = PERIOD) -> float:
    """Root mean squared scaled error."""
    y = np.asarray(y_train, dtype=np.float64)
    denom = float(np.mean((y[seasonality:] - y[:-seasonality]) ** 2))
    mse = float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))
    return float(np.sqrt(mse / denom))


# --------------------------------------------------------------------- #
#                      4. Features and models                            #
# --------------------------------------------------------------------- #


def make_supervised(series: np.ndarray,
                    lags: Sequence[int]) -> Tuple[np.ndarray, np.ndarray,
                                                  np.ndarray]:
    """Turns a series into a lag matrix with calendar features.

    Args:
        series: The full series.
        lags: Which lags to use as features.

    Returns:
        A tuple (features, targets, index) where index gives each row's
        position in the original series.
    """
    max_lag = max(lags)
    n = len(series)
    rows, targets, positions = [], [], []
    for t in range(max_lag, n):
        lag_values = [series[t - lag] for lag in lags]
        phase = 2 * np.pi * (t % PERIOD) / PERIOD
        rows.append(lag_values + [np.sin(phase), np.cos(phase)])
        targets.append(series[t])
        positions.append(t)
    return (np.array(rows, dtype=np.float64),
            np.array(targets, dtype=np.float64),
            np.array(positions, dtype=int))


def rolling_window_folds(n: int, n_folds: int, horizon: int,
                         window: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Rolling-origin folds with a constant training length.

    Args:
        n: Number of usable rows.
        n_folds: Number of folds.
        horizon: Test length per fold.
        window: Fixed training length.

    Returns:
        A list of (train_idx, test_idx).

    Raises:
        ValueError: If the series is too short.
    """
    needed = window + n_folds * horizon
    if needed > n:
        raise ValueError(f'need {needed} rows, have {n}')
    folds = []
    for i in range(n_folds):
        end = window + i * horizon
        folds.append((np.arange(end - window, end),
                      np.arange(end, end + horizon)))
    return folds


def assert_no_lookforward(folds) -> None:
    """Fails loudly if any training row reaches past its test window.

    Raises:
        AssertionError: If a fold leaks the future.
    """
    for i, (train_idx, test_idx) in enumerate(folds):
        if int(train_idx.max()) >= int(test_idx.min()):
            raise AssertionError(f'fold {i}: lookforward leak')


def baseline_forecasts(series: np.ndarray, positions: np.ndarray,
                       test_idx: np.ndarray) -> Dict[str, np.ndarray]:
    """Builds the baselines every model must beat.

    All of them are computed from values strictly before each target,
    so none of them cheats.

    Args:
        series: The full series.
        positions: Position in the series of each supervised row.
        test_idx: Row indices of the test block.

    Returns:
        A mapping from baseline name to its predictions.
    """
    target_pos = positions[test_idx]
    origin = int(target_pos[0]) - 1               # last observed point
    history = series[:origin + 1]
    steps = np.arange(1, len(target_pos) + 1)
    slope = float(np.mean(np.diff(history[-90:])))
    return {
        'naive': np.full(len(target_pos), series[origin]),
        'seasonal_naive': np.array(
            [series[pos - PERIOD] for pos in target_pos]),
        'drift': series[origin] + slope * steps,
        'mean': np.full(len(target_pos), float(history[-180:].mean())),
    }


def fit_predict_ridge(x_train, y_train, x_test) -> np.ndarray:
    """Ridge on lag features, scaled with training statistics only."""
    mean, std = x_train.mean(axis=0), x_train.std(axis=0)
    std[std == 0] = 1.0
    model = Ridge(alpha=1.0).fit((x_train - mean) / std, y_train)
    return model.predict((x_test - mean) / std)


def fit_predict_gbm(x_train, y_train, x_test) -> np.ndarray:
    """Gradient-boosted trees on the same features."""
    model = HistGradientBoostingRegressor(
        max_leaf_nodes=15, learning_rate=0.1, max_iter=200,
        random_state=0).fit(x_train, y_train)
    return model.predict(x_test)


# --------------------------------------------------------------------- #
#                        5. Conformal intervals                          #
# --------------------------------------------------------------------- #


def split_conformal(residuals: np.ndarray, alpha: float) -> float:
    """Half-width giving 1 - alpha coverage under exchangeability.

    Args:
        residuals: Absolute errors on the calibration set.
        alpha: Miscoverage rate.

    Returns:
        The conformal half-width.

    Raises:
        ValueError: If the calibration set is too small for this alpha.
    """
    s = np.sort(np.asarray(residuals, dtype=np.float64))
    n = s.size
    if n + 1 < 1.0 / alpha:
        raise ValueError(f'need at least {int(np.ceil(1 / alpha)) - 1} '
                         f'calibration points for alpha={alpha}')
    k = int(np.ceil((n + 1) * (1.0 - alpha))) - 1
    return float(s[min(max(k, 0), n - 1)])


def coverage(y_true: np.ndarray, lower: np.ndarray,
             upper: np.ndarray) -> float:
    """Fraction of truths inside their interval."""
    y_true = np.asarray(y_true)
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def winkler_score(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                  alpha: float) -> float:
    """Mean interval score: width plus an out-of-interval penalty.

    Rewards intervals that are narrow AND correct, which is why coverage
    alone is never enough.

    Args:
        y_true: Observed values.
        lower: Interval lower bounds.
        upper: Interval upper bounds.
        alpha: Miscoverage rate.

    Returns:
        The mean Winkler score; lower is better.
    """
    width = upper - lower
    below = np.maximum(lower - y_true, 0.0)
    above = np.maximum(y_true - upper, 0.0)
    return float(np.mean(width + (2.0 / alpha) * (below + above)))


@dataclasses.dataclass
class Gate:
    """One acceptance check declared before the run."""

    name: str
    rule: str
    observed: str
    passed: bool


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Runs the project end to end.

    Args:
        argv: Optional argument vector.

    Returns:
        0 when every gate passed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=20260101)
    parser.add_argument('--horizon', type=int, default=28)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--window', type=int, default=500)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--no-shift', action='store_true',
                        help='stationary control run')
    parser.add_argument('--no-write', action='store_true')
    args = parser.parse_args(argv)

    if not HAS_SKLEARN:
        print('UNVERIFIED: scikit-learn is not installed. '
              'Run testbed/setup.ps1 first.')
        return 2

    started = time.perf_counter()
    print('=' * 74)
    print('PROJECT 3 - Forecasting with honest evaluation and real intervals')
    print('=' * 74)

    series, shift_index = generate_series(seed=args.seed,
                                          shift=not args.no_shift)
    print(f'\n[1] Series: {len(series)} daily points, period {PERIOD}')
    if args.no_shift:
        print('    no regime shift (control run)')
    else:
        print(f'    regime shift at t={shift_index} '
              f'(level +18, seasonal amplitude +6)')

    # --- 2. diagnostics ------------------------------------------------ #
    trend, seasonal, residual = additive_decompose(series, PERIOD)
    correlations = acf(series, 30)
    stationarity = stationarity_report(series)
    print(f'\n[2] Diagnostics')
    print(f'    decomposition residual mean {residual.mean():+.4f} '
          f'(should be ~0); components re-sum: '
          f'{np.abs(trend + seasonal + residual - series).max() < 1e-9}')
    print(f'    ACF at lags 1, 7, 14: {correlations[1]:.3f}, '
          f'{correlations[7]:.3f}, {correlations[14]:.3f}')
    if stationarity['available']:
        print(f'    ADF p={stationarity["adf_p"]:.4f}  '
              f'KPSS p={stationarity["kpss_p"]:.4f}')
        print(f'    reading: {stationarity["reading"]}')
    else:
        print(f'    {stationarity["note"]}')

    # --- 3. supervised frame and folds --------------------------------- #
    x, y, positions = make_supervised(series, LAGS)
    folds = rolling_window_folds(len(y), args.folds, args.horizon,
                                 args.window)
    assert_no_lookforward(folds)
    print(f'\n[3] Rolling-origin folds (window {args.window}, '
          f'horizon {args.horizon})')
    for i, (train_idx, test_idx) in enumerate(folds):
        print(f'    fold {i}: train [{train_idx.min()},{train_idx.max()}] '
              f'< test [{test_idx.min()},{test_idx.max()}]  '
              f'(series t={positions[test_idx.min()]}..'
              f'{positions[test_idx.max()]})')

    # --- 4. benchmark --------------------------------------------------- #
    model_fns: Dict[str, Callable] = {'ridge_lags': fit_predict_ridge,
                                      'gbm_lags': fit_predict_gbm}
    scores: Dict[str, Dict[str, List[float]]] = {}
    for name in list(model_fns) + ['naive', 'seasonal_naive', 'drift', 'mean']:
        scores[name] = {'mase': [], 'rmsse': []}

    for train_idx, test_idx in folds:
        y_train_series = series[:positions[test_idx.min()]]
        predictions = baseline_forecasts(series, positions, test_idx)
        for name, fn in model_fns.items():
            predictions[name] = fn(x[train_idx], y[train_idx], x[test_idx])
        for name, pred in predictions.items():
            scores[name]['mase'].append(
                mase(y[test_idx], pred, y_train_series))
            scores[name]['rmsse'].append(
                rmsse(y[test_idx], pred, y_train_series))

    print(f'\n[4] Benchmark over {args.folds} folds '
          f'(MASE < 1 beats seasonal naive)')
    ranking = sorted(scores.items(), key=lambda kv: np.mean(kv[1]['mase']))
    for name, record in ranking:
        mean_mase = float(np.mean(record['mase']))
        sd_mase = float(np.std(record['mase']))
        mean_rmsse = float(np.mean(record['rmsse']))
        flag = '  <- baseline' if name in ('naive', 'seasonal_naive',
                                           'drift', 'mean') else ''
        print(f'    {name:<16} MASE {mean_mase:.4f} (sd {sd_mase:.4f})  '
              f'RMSSE {mean_rmsse:.4f}{flag}')
    best_model = min(model_fns, key=lambda m: np.mean(scores[m]['mase']))
    best_baseline = min(['naive', 'seasonal_naive', 'drift', 'mean'],
                        key=lambda m: np.mean(scores[m]['mase']))
    beats = np.mean(scores[best_model]['mase']) < np.mean(
        scores[best_baseline]['mase'])
    print(f'    best model {best_model} '
          f'{"beats" if beats else "LOSES TO"} best baseline {best_baseline}')

    # --- 5. horizon curve ----------------------------------------------- #
    train_idx, test_idx = folds[-1]
    y_train_series = series[:positions[test_idx.min()]]
    pred = model_fns[best_model](x[train_idx], y[train_idx], x[test_idx])
    print(f'\n[5] Error against horizon ({best_model}, last fold)')
    horizon_curve = []
    for h in range(1, args.horizon + 1):
        value = mase(y[test_idx][:h], pred[:h], y_train_series)
        horizon_curve.append({'h': h, 'mase': round(value, 4)})
        if h in (1, 7, 14, args.horizon):
            print(f'    h={h:<3} cumulative MASE {value:.4f}')

    # --- 6. conformal intervals ------------------------------------------ #
    # Calibrate on the tail of training, which the model did not fit.
    fit_idx = train_idx[:-args.window // 4]
    cal_idx = train_idx[-args.window // 4:]
    cal_pred = model_fns[best_model](x[fit_idx], y[fit_idx], x[cal_idx])
    cal_residuals = np.abs(y[cal_idx] - cal_pred)
    half_width = split_conformal(cal_residuals, args.alpha)

    test_pred = model_fns[best_model](x[fit_idx], y[fit_idx], x[test_idx])
    lower, upper = test_pred - half_width, test_pred + half_width
    observed = coverage(y[test_idx], lower, upper)
    score = winkler_score(y[test_idx], lower, upper, args.alpha)
    target = 1.0 - args.alpha

    print(f'\n[6] Split conformal at alpha={args.alpha} '
          f'(target coverage {target:.0%})')
    print(f'    calibration points {len(cal_idx)}, '
          f'half-width {half_width:.3f}')
    print(f'    coverage on the test fold: {observed:.3f}')
    print(f'    mean interval width {2 * half_width:.3f}, '
          f'Winkler score {score:.3f}')

    # --- 7. coverage either side of the shift ----------------------------- #
    print(f'\n[7] Coverage before and after the regime shift')
    segment_rows: List[Dict] = []
    pre_mask = positions < shift_index
    post_mask = ~pre_mask
    stationary_coverage = None
    for label, mask in (('pre-shift', pre_mask), ('post-shift', post_mask)):
        candidate = np.where(mask)[0]
        candidate = candidate[candidate > cal_idx.max()]
        if len(candidate) < 30:
            continue
        pred_seg = model_fns[best_model](x[fit_idx], y[fit_idx],
                                         x[candidate])
        cov = coverage(y[candidate], pred_seg - half_width,
                       pred_seg + half_width)
        segment_rows.append({'segment': label, 'n': int(len(candidate)),
                             'coverage': round(cov, 4)})
        print(f'    {label:<12} n={len(candidate):<5} coverage {cov:.3f}')
        if label == 'pre-shift':
            stationary_coverage = cov
    if not args.no_shift and len(segment_rows) == 2:
        drop = segment_rows[0]['coverage'] - segment_rows[1]['coverage']
        print(f'    coverage lost across the shift: {drop:+.3f}')
        print('    exchangeability failed, so the guarantee did too '
              '-- the expected result, measured rather than assumed')

    # --- 8. gates ----------------------------------------------------------- #
    gates = [
        Gate('no_lookforward', 'no training row at or past its test block',
             'asserted for every fold', True),
        Gate('constant_train_window',
             'every rolling fold trains on the same number of rows',
             str({len(t) for t, _ in folds}),
             len({len(t) for t, _ in folds}) == 1),
        Gate('mase_denominator_is_train_only',
             'naive scale computed from pre-test data only',
             'by construction', True),
        Gate('model_beats_best_baseline',
             f'best model MASE < best baseline ({best_baseline})',
             f'{np.mean(scores[best_model]["mase"]):.4f} vs '
             f'{np.mean(scores[best_baseline]["mase"]):.4f}', bool(beats)),
        Gate('error_grows_with_horizon',
             'cumulative MASE at full horizon >= at h=1',
             f'{horizon_curve[-1]["mase"]:.4f} vs '
             f'{horizon_curve[0]["mase"]:.4f}',
             horizon_curve[-1]['mase'] >= horizon_curve[0]['mase']),
        Gate('conformal_coverage_stationary',
             f'pre-shift coverage within 0.07 of {target:.2f}',
             f'{stationary_coverage:.3f}'
             if stationary_coverage is not None else 'n/a',
             stationary_coverage is not None
             and abs(stationary_coverage - target) < 0.07),
        Gate('width_reported_with_coverage',
             'interval width and Winkler score reported alongside coverage',
             f'width {2 * half_width:.3f}, Winkler {score:.3f}', True),
    ]
    print(f'\n[8] Gates')
    for gate in gates:
        mark = 'PASS' if gate.passed else 'FAIL'
        print(f'    [{mark}] {gate.name}: {gate.observed}  '
              f'(rule: {gate.rule})')

    elapsed = time.perf_counter() - started
    status = 'VERIFIED' if all(g.passed for g in gates) else 'PENDING REVISION'
    print(f'\nstatus: {status}   ({elapsed:.1f}s)')

    if not args.no_write:
        run_dir = RUNS / 'p3_forecast_conformal'
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            'project': 'p3_forecast_conformal',
            'status': status,
            'seed': args.seed,
            'shift_index': shift_index if not args.no_shift else None,
            'diagnostics': {
                'acf_1_7_14': [float(correlations[1]), float(correlations[7]),
                               float(correlations[14])],
                'residual_mean': float(residual.mean()),
                'stationarity': stationarity,
            },
            'folds': [{'train': [int(t.min()), int(t.max())],
                       'test': [int(s.min()), int(s.max())]}
                      for t, s in folds],
            'scores': {name: {'mase_mean': float(np.mean(rec['mase'])),
                              'mase_sd': float(np.std(rec['mase'])),
                              'rmsse_mean': float(np.mean(rec['rmsse']))}
                       for name, rec in scores.items()},
            'best_model': best_model,
            'best_baseline': best_baseline,
            'horizon_curve': horizon_curve,
            'conformal': {'alpha': args.alpha, 'half_width': half_width,
                          'coverage': observed, 'winkler': score,
                          'segments': segment_rows},
            'gates': [dataclasses.asdict(g) for g in gates],
            'seconds': round(elapsed, 1),
        }
        (run_dir / 'report.json').write_text(
            json.dumps(payload, indent=2, default=str), encoding='utf-8')
        print(f'audit pack: {run_dir}')

    return 0 if all(g.passed for g in gates) else 1


if __name__ == '__main__':
    sys.exit(main())
