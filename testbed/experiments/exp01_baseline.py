r"""EXP-01: the single baseline experiment.

Guideline 5 of the Prompt Guidelines says to start with one document and one
baseline experiment. This is that experiment, and it is the template for
every later one.

It fits the same linear model by several methodologies, scores them all with
one metrics module on byte-identical data, and checks them against values
that are known in advance rather than against each other:

    normal_equations  closed form least squares   [MML p.210]
    gradient_descent  hand-written, full batch    [MML p.381, p.481]
    sklearn_linear    library reference           [SKL ch.4]
    torch_sgd         mini-batch autograd         [PTG p.45, p.62]

Arms whose dependency is missing report UNVERIFIED rather than being
silently dropped (ledger GAP-13).

Usage:
    python -m experiments.exp01_baseline --config configs/exp01.toml
    python -m experiments.exp01_baseline --config configs/exp01.toml \\
        --sweep arms.gradient_descent.lr=0.01,0.05,0.5,1.2
"""

import inspect
import pathlib
import sys
import time
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from mlkit import STATUS_BUG  # noqa: E402  (path set above)
from mlkit import STATUS_UNVERIFIED  # noqa: E402
from mlkit import STATUS_VERIFIED  # noqa: E402
from mlkit import config as config_lib  # noqa: E402
from mlkit import data as data_lib  # noqa: E402
from mlkit import metrics  # noqa: E402
from mlkit import report as report_lib  # noqa: E402
from mlkit import splits  # noqa: E402

EXPERIMENT = 'exp01_baseline'

ANCHORS: Dict[str, str] = {
    'normal_equations': '[MML p.210] orthogonal projection; [MML p.619] MLE',
    'gradient_descent': '[MML p.381] basics; [MML p.481] full form',
    'sklearn_linear': '[SKL ch.4 "Linear and Generalized Linear Models"]',
    'torch_sgd': '[PTG p.45] computation graph; [PTG p.62] training loop',
}

Weights = np.ndarray


def add_bias(x: np.ndarray) -> np.ndarray:
    """Appends a column of ones so the intercept is learned like a weight.

    Args:
        x: Feature matrix of shape (n, d).

    Returns:
        A matrix of shape (n, d + 1).
    """
    return np.hstack([x, np.ones((x.shape[0], 1), dtype=x.dtype)])


def fit_normal_equations(
    x_train: np.ndarray,
    y_train: np.ndarray,
    **_: Any,
) -> Weights:
    """Solves least squares in closed form.

    This is the orthogonal projection of y onto the column space of X
    [MML p.210], and the maximum-likelihood estimate under Gaussian noise
    [MML p.619]. It is the reference the other arms are judged against.

    Args:
        x_train: Training features, shape (n, d).
        y_train: Training targets, shape (n,).

    Returns:
        The weight vector including the intercept as its last element.
    """
    design = add_bias(x_train)
    gram = design.T @ design
    moment = design.T @ y_train
    return np.linalg.solve(gram, moment)


def fit_gradient_descent(
    x_train: np.ndarray,
    y_train: np.ndarray,
    lr: float = 0.05,
    iters: int = 2000,
    momentum: float = 0.0,
    **_: Any,
) -> Weights:
    """Fits the same model by full-batch gradient descent.

    The update is the one derived at [MML p.381] and generalised to many
    variables at [MML p.481]. The loss is the mean squared error, so the
    gradient is (2/n) X^T (Xw - y).

    Args:
        x_train: Training features, shape (n, d).
        y_train: Training targets, shape (n,).
        lr: Learning rate. Divergence above 2/L is expected, not a bug.
        iters: Number of full-batch steps.
        momentum: Heavy-ball coefficient; 0.0 gives plain descent.

    Returns:
        The weight vector including the intercept as its last element.
    """
    design = add_bias(x_train)
    n_samples = design.shape[0]
    weights = np.zeros(design.shape[1], dtype=np.float64)
    velocity = np.zeros_like(weights)
    for _ in range(iters):
        residual = design @ weights - y_train
        gradient = (2.0 / n_samples) * (design.T @ residual)
        velocity = momentum * velocity - lr * gradient
        weights = weights + velocity
        if not np.all(np.isfinite(weights)):
            break
    return weights


def fit_sklearn_linear(
    x_train: np.ndarray,
    y_train: np.ndarray,
    **_: Any,
) -> Weights:
    """Fits the library reference implementation.

    Args:
        x_train: Training features, shape (n, d).
        y_train: Training targets, shape (n,).

    Returns:
        The weight vector including the intercept as its last element.

    Raises:
        ImportError: If scikit-learn is not installed.
    """
    from sklearn.linear_model import LinearRegression  # noqa: PLC0415

    model = LinearRegression().fit(x_train, y_train)
    return np.append(model.coef_, model.intercept_)


def fit_torch_sgd(
    x_train: np.ndarray,
    y_train: np.ndarray,
    lr: float = 0.05,
    epochs: int = 200,
    batch_size: int = 32,
    seed: int = 20260101,
    **_: Any,
) -> Weights:
    """Fits the model with mini-batch SGD and autograd.

    The loop is deliberately written out rather than wrapped, matching the
    structure at [PTG p.62] so the computation graph stays visible.

    Args:
        x_train: Training features, shape (n, d).
        y_train: Training targets, shape (n,).
        lr: Learning rate.
        epochs: Number of passes over the data.
        batch_size: Mini-batch size [PTG p.72].
        seed: Seed for the shuffling generator.

    Returns:
        The weight vector including the intercept as its last element.

    Raises:
        ImportError: If PyTorch is not installed.
    """
    import torch  # noqa: PLC0415

    torch.manual_seed(seed)
    features = torch.tensor(x_train, dtype=torch.float32)
    targets = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    model = torch.nn.Linear(features.shape[1], 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    n_samples = features.shape[0]
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(n_samples, generator=generator)
        for start in range(0, n_samples, batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(features[batch]), targets[batch])
            loss.backward()
            optimizer.step()
    weights = model.weight.detach().numpy().ravel()
    bias = model.bias.detach().numpy().ravel()
    return np.append(weights, bias)


ARMS: Dict[str, Callable[..., Weights]] = {
    'normal_equations': fit_normal_equations,
    'gradient_descent': fit_gradient_descent,
    'sklearn_linear': fit_sklearn_linear,
    'torch_sgd': fit_torch_sgd,
}


def predict(weights: Weights, x: np.ndarray) -> np.ndarray:
    """Applies a weight vector that carries its intercept last."""
    return add_bias(x) @ weights


def score_arm(
    name: str,
    weights: Optional[Weights],
    x_test: np.ndarray,
    y_test: np.ndarray,
    params: Dict[str, Any],
    seconds: float,
    n_boot: int,
    level: float,
    seed: int,
    note: str = '',
    failure_status: str = STATUS_UNVERIFIED,
) -> report_lib.ArmResult:
    """Scores one arm with the single measurement tools.

    Args:
        name: Arm name.
        weights: Fitted weights, or None when the arm could not run.
        x_test: Held-out features.
        y_test: Held-out targets.
        params: Parameters the arm ran with.
        seconds: Wall-clock time for the fit.
        n_boot: Bootstrap resamples for the confidence interval.
        level: Confidence level.
        seed: Seed for the bootstrap.
        note: Explanation attached to the result.
        failure_status: Tag to use when the arm did not produce weights --
            UNVERIFIED for a missing dependency, BUG for one that is
            installed but broken.

    Returns:
        The arm result, carrying `failure_status` when the arm did not run.
    """
    if weights is None:
        return report_lib.ArmResult(
            name=name,
            params=params,
            status=failure_status,
            notes=note,
        )
    predictions = predict(weights, x_test)
    values = {
        key: fn(y_test, predictions)
        for key, fn in metrics.REGRESSION_METRICS.items()
    }
    lower, upper = metrics.bootstrap_ci(
        metrics.rmse,
        y_test,
        predictions,
        n_boot=n_boot,
        level=level,
        seed=seed,
    )
    return report_lib.ArmResult(
        name=name,
        params=params,
        metrics=values,
        ci={'rmse': [lower, upper]},
        seconds=seconds,
        notes=note,
    )


def run_experiment(config: Dict[str, Any]) -> report_lib.RunReport:
    """Runs every configured arm and applies the acceptance gates.

    Args:
        config: Resolved experiment configuration.

    Returns:
        The run report, with each arm tagged and each gate recorded.
    """
    seed = int(config['seed'])
    dataset_cfg = dict(config['dataset'])
    name = dataset_cfg.pop('name')
    x, y, card, extras = data_lib.load_dataset(name, seed=seed, **dataset_cfg)

    fold = splits.random_split(
        n_samples=x.shape[0],
        test_size=float(config['split']['test_size']),
        seed=seed,
    )
    splits.assert_disjoint([fold])
    x_train, y_train = x[fold.train_idx], y[fold.train_idx]
    x_test, y_test = x[fold.test_idx], y[fold.test_idx]

    report = report_lib.RunReport(
        experiment=EXPERIMENT,
        seed=seed,
        config=config,
        dataset_card=card.as_dict(),
        splits=list(splits.summarise([fold])),
        gaps=['GAP-12: runtime unverified on this hardware'],
    )

    fitted: Dict[str, Weights] = {}
    report_cfg = config.get('report', {})
    n_boot = int(report_cfg.get('bootstrap', 1000))
    level = float(report_cfg.get('ci', 0.95))

    for arm_name, params in config['arms'].items():
        if arm_name not in ARMS:
            raise KeyError(f'unknown arm {arm_name!r}')
        kwargs = dict(params)
        kwargs.setdefault('seed', seed)
        started = time.perf_counter()
        failure_status = STATUS_UNVERIFIED
        note = ''
        try:
            weights = ARMS[arm_name](x_train, y_train, **kwargs)
        except ImportError as error:
            weights = None
            note = (
                f'dependency missing ({error.name}); arm tagged '
                f'{STATUS_UNVERIFIED} -- install it to close GAP-13'
            )
        # A dependency that is installed but broken is a different thing
        # from one that is absent, and it must not be allowed to look like
        # a clean skip. We catch it, tag it BUG, and keep the diagnosis.
        except Exception as error:  # pylint: disable=broad-except
            weights = None
            failure_status = STATUS_BUG
            note = (
                f'{type(error).__name__}: {error} -- the dependency is '
                f'installed but broken; log it in VERIFICATION.md'
            )
        elapsed = time.perf_counter() - started
        if weights is not None:
            fitted[arm_name] = weights
        report.arms.append(
            score_arm(
                arm_name,
                weights,
                x_test,
                y_test,
                dict(params),
                elapsed,
                n_boot,
                level,
                seed,
                note,
                failure_status,
            )
        )

    _apply_gates(report, fitted, extras, x_test, y_test, config)
    report.resolve_status()
    return report


def _apply_gates(
    report: report_lib.RunReport,
    fitted: Dict[str, Weights],
    extras: Dict[str, Any],
    x_test: np.ndarray,
    y_test: np.ndarray,
    config: Dict[str, Any],
) -> None:
    """Applies the acceptance gates declared before the code was written.

    Args:
        report: Report to append gates to, modified in place.
        fitted: Weight vectors by arm name.
        extras: Ground truth from the data generator.
        x_test: Held-out features.
        y_test: Held-out targets.
        config: Resolved configuration, for the declared tolerances.
    """
    gates_cfg = config.get('gates', {})
    truth = np.append(extras['true_coef'], extras['true_bias'])
    noise = float(extras['noise'])
    passed_by_arm: Dict[str, bool] = {}

    reference = fitted.get('normal_equations')
    if reference is not None:
        recovery_tol = float(gates_cfg.get('coef_recovery_tol', 0.1))
        error = float(np.max(np.abs(reference - truth)))
        floor_lo, floor_hi = 0.8 * noise, 1.3 * noise
        observed_rmse = metrics.rmse(y_test, predict(reference, x_test))
        checks = [
            report_lib.Gate(
                name='closed_form_recovers_truth',
                anchor='[MML p.210], [MML p.619]',
                threshold=f'max |w - w_true| < {recovery_tol}',
                observed=f'{error:.6g}',
                passed=error < recovery_tol,
            ),
            report_lib.Gate(
                name='rmse_at_irreducible_error',
                anchor='[MML p.619] Gaussian MLE; noise sd on the card',
                threshold=f'{floor_lo:.4g} <= rmse <= {floor_hi:.4g}',
                observed=f'{observed_rmse:.6g}',
                passed=floor_lo <= observed_rmse <= floor_hi,
            ),
        ]
        report.gates.extend(checks)
        passed_by_arm['normal_equations'] = all(c.passed for c in checks)

    # Every other arm is judged against the closed-form reference, each at
    # the tolerance its arithmetic allows: float64 descent must agree to
    # 1e-6, float32 mini-batch SGD only to a looser bound.
    comparisons = (
        (
            'gradient_descent',
            'gd_agreement_tol',
            1e-6,
            '[MML p.381], [MML p.481]',
        ),
        (
            'sklearn_linear',
            'sklearn_agreement_tol',
            1e-6,
            '[SKL ch.4 "Linear and Generalized Linear Models"]',
        ),
        ('torch_sgd', 'torch_agreement_tol', 5e-2, '[PTG p.45], [PTG p.62]'),
    )
    for arm_name, key, default_tol, anchor in comparisons:
        candidate = fitted.get(arm_name)
        if reference is None or candidate is None:
            continue
        tol = float(gates_cfg.get(key, default_tol))
        gap = float(np.max(np.abs(candidate - reference)))
        gate = report_lib.Gate(
            name=f'{arm_name}_matches_closed_form',
            anchor=anchor,
            threshold=f'max |w - w_closed| < {tol}',
            observed=f'{gap:.3g}',
            passed=bool(np.isfinite(gap) and gap < tol),
        )
        report.gates.append(gate)
        passed_by_arm[arm_name] = gate.passed

    for arm in report.arms:
        if arm.status == STATUS_UNVERIFIED:
            continue
        if passed_by_arm.get(arm.name):
            arm.status = STATUS_VERIFIED


def print_comparison(report: report_lib.RunReport, full: bool = False) -> None:
    """Prints the cross-methodology comparison table.

    Args:
        report: The report to render.
        full: When True, also print every arm's secondary metrics. The
            short form is the default because the primary metric plus its
            interval is what a decision is actually made on.
    """
    print(f'\n{"arm":<20}{"status":<18}{"rmse":>12}{"95% CI":>26}{"s":>9}')
    print('-' * 85)
    for arm in report.arms:
        rmse_value = arm.metrics.get('rmse')
        interval = arm.ci.get('rmse')
        rmse_text = f'{rmse_value:.6f}' if rmse_value is not None else '--'
        ci_text = (
            f'[{interval[0]:.4f}, {interval[1]:.4f}]' if interval else '--'
        )
        print(
            f'{arm.name:<20}{arm.status:<18}{rmse_text:>12}'
            f'{ci_text:>26}{arm.seconds:>9.3f}'
        )
        if full and arm.metrics:
            secondary = ', '.join(
                f'{key}={value:.6g}'
                for key, value in sorted(arm.metrics.items())
                if key != 'rmse'
            )
            print(f'{"":<20}{secondary}')
    print('\ngates')
    for gate in report.gates:
        mark = 'PASS' if gate.passed else 'FAIL'
        print(f'  [{mark}] {gate.name}: {gate.observed} vs {gate.threshold}')
        print(f'         anchor: {gate.anchor}')
    for arm in report.arms:
        if arm.notes:
            print(f'  note ({arm.name}): {arm.notes}')
    print(f'\nEXP-01 overall status: {report.status}')


def run_sweep(config: Dict[str, Any], spec: str) -> None:
    """Runs one parameter range and prints the parameter/metric curve.

    The divergence of gradient descent above lr = 2/L is a required result
    of this program, not a failure [MML p.381].

    Args:
        config: Resolved configuration.
        spec: A sweep specification such as 'arms.gradient_descent.lr=0.1,1.5'.
    """
    key, values = config_lib.parse_sweep(spec)
    arm_name = key.split('.')[1] if key.startswith('arms.') else None
    print(f'\nsweep {key}')
    print(f'{"value":>12}{"rmse":>24}{"status":>20}')
    print('-' * 56)
    for value in values:
        swept = config_lib.apply_overrides(config, [f'{key}={value}'])
        # Divergence is an expected outcome of this sweep [MML p.381], so
        # overflow is data here, not an error to print at the user.
        with np.errstate(all='ignore'):
            report = run_experiment(swept)
        match = [a for a in report.arms if a.name == arm_name]
        arm = match[0] if match else report.arms[0]
        rmse_value = arm.metrics.get('rmse')
        if rmse_value is None or not np.isfinite(rmse_value):
            shown = 'diverged'
        elif abs(rmse_value) > 1e6:
            shown = f'diverged ({rmse_value:.2e})'
        else:
            shown = f'{rmse_value:.6f}'
        print(f'{str(value):>12}{shown:>24}{report.status:>20}')
    print(
        '\nDivergence above the stability limit is a required result of '
        'this sweep,\nnot a failure: see [MML p.381] and [MML p.389].'
    )


def show_source(arm_name: str) -> None:
    """Prints an arm's implementation and its source anchors.

    Args:
        arm_name: Name of the arm to show.

    Raises:
        KeyError: If the arm is unknown.
    """
    if arm_name not in ARMS:
        raise KeyError(f'unknown arm {arm_name!r}; have {sorted(ARMS)}')
    print(f'# anchors: {ANCHORS[arm_name]}\n')
    print(inspect.getsource(ARMS[arm_name]))


def main(argv: Optional[Tuple[str, ...]] = None) -> int:
    """Entry point for the baseline experiment.

    Args:
        argv: Optional argument vector, for testing.

    Returns:
        0 when the run is verified, 1 otherwise, so CI can gate on it.
    """
    parser = config_lib.build_parser(__doc__ or EXPERIMENT)
    args = parser.parse_args(argv)

    if args.show_source:
        show_source(args.show_source)
        return 0

    config = config_lib.apply_overrides(
        config_lib.load_config(args.config), args.overrides
    )

    if args.sweep:
        run_sweep(config, args.sweep)
        return 0

    report = run_experiment(config)
    print_comparison(report, full=args.compare)
    if not args.no_write:
        run_dir = report_lib.write_run(report)
        print(f'audit pack: {run_dir}')
    return 0 if report.status == STATUS_VERIFIED else 1


if __name__ == '__main__':
    sys.exit(main())
