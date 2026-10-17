"""Stage 1 acceptance gates.

These compare your implementations in `exercises/stage1_linalg.py` against
NumPy and against identities that hold independently of any implementation.

A gate skips while its function is unimplemented, so the rest of the
testbed stays green. A skip is not a pass: the competency row in
VERIFICATION.md stays at '--' until the gate actually runs.

Run just this file:
    pytest tests/test_stage1_gates.py -v
"""

import numpy as np
import pytest

from exercises import stage1_linalg as s1


def _require(fn, *args, **kwargs):
    """Runs a Stage 1 function, skipping while it is unimplemented."""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as error:
        pytest.skip(f'not implemented yet: {error}')
        raise  # pragma: no cover -- pytest.skip raises


def test_gate_lp_norm_matches_numpy():
    """M1.1 gate: norms agree with NumPy to 1e-12 [MML p.49]."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=64)
    for order in (1.0, 2.0, float('inf')):
        got = _require(s1.lp_norm, x, order)
        expected = float(np.linalg.norm(x, ord=order))
        assert got == pytest.approx(expected, rel=1e-12)
    assert _require(s1.lp_norm, np.array([3.0, 4.0]), 2.0) == pytest.approx(5.0)


def test_gate_gram_schmidt_produces_orthonormal_columns():
    """M1.1 gate: Q^T Q = I to 1e-10 [MML p.67, p.89]."""
    rng = np.random.default_rng(1)
    a = rng.normal(size=(50, 6))
    q = _require(s1.gram_schmidt, a)
    assert q.shape == a.shape
    assert np.allclose(q.T @ q, np.eye(a.shape[1]), atol=1e-10)


def test_gate_lu_reconstructs_the_matrix():
    """M1.2 gate: ||PA - LU|| < 1e-10 [MML p.171]."""
    rng = np.random.default_rng(2)
    a = rng.normal(size=(8, 8))
    p, lower, upper = _require(s1.lu_decompose, a)
    assert np.allclose(p @ a, lower @ upper, atol=1e-10)
    assert np.allclose(np.tril(lower), lower, atol=1e-12)
    assert np.allclose(np.triu(upper), upper, atol=1e-12)


def test_gate_power_iteration_finds_the_dominant_eigenpair():
    """M1.3 gate: matches `np.linalg.eigh` [MML p.217]."""
    rng = np.random.default_rng(3)
    base = rng.normal(size=(20, 20))
    a = base @ base.T  # symmetric positive semi-definite
    value, vector = _require(s1.power_iteration, a)
    expected = float(np.linalg.eigvalsh(a).max())
    assert value == pytest.approx(expected, rel=1e-6)
    assert np.allclose(a @ vector, value * vector, atol=1e-6 * abs(value))


def test_gate_svd_lowrank_hits_the_eckart_young_error():
    """M1.3 gate: the error identity holds exactly [MML p.207]."""
    rng = np.random.default_rng(4)
    a = rng.normal(size=(30, 12))
    k = 5
    approx = _require(s1.svd_lowrank, a, k)
    singular = np.linalg.svd(a, compute_uv=False)
    expected = float(np.sqrt(np.sum(singular[k:] ** 2)))
    assert np.linalg.norm(a - approx) == pytest.approx(expected, abs=1e-10)


def test_gate_gradient_descent_converges_below_the_stability_limit():
    """M1.4 gate: convergence for lr < 2/L [MML p.381]."""
    curvature = np.array([4.0, 1.0])  # L = 2 * max(curvature) = 8

    def grad(x):
        return 2.0 * curvature * x

    limit = 2.0 / (2.0 * curvature.max())
    x0 = np.array([5.0, -3.0])
    converged = _require(s1.gradient_descent, grad, x0, 0.5 * limit, 2000)
    assert np.linalg.norm(converged) < 1e-6


def test_gate_gradient_descent_diverges_above_the_stability_limit():
    """M1.4 gate: divergence above 2/L is a required observation.

    A gradient descent that never diverges has been silently clamped, and
    the lesson at [MML p.389] has been hidden rather than learned.
    """
    curvature = np.array([4.0, 1.0])

    def grad(x):
        return 2.0 * curvature * x

    limit = 2.0 / (2.0 * curvature.max())
    x0 = np.array([5.0, -3.0])
    diverged = _require(s1.gradient_descent, grad, x0, 1.5 * limit, 200)
    assert not np.all(np.isfinite(diverged)) or np.linalg.norm(diverged) > 1e3


def test_gate_numeric_jacobian_matches_the_analytic_one():
    """M1.5 gate: agreement to 1e-6 [MML p.460]."""

    def f(x):
        return np.array([x[0] ** 2 + x[1], np.sin(x[2]) * x[0]])

    def analytic(x):
        return np.array(
            [
                [2.0 * x[0], 1.0, 0.0],
                [np.sin(x[2]), 0.0, np.cos(x[2]) * x[0]],
            ]
        )

    x = np.array([0.7, -1.3, 0.4])
    got = _require(s1.numeric_jacobian, f, x)
    assert np.allclose(got, analytic(x), atol=1e-6)


def test_gate_entropy_is_maximal_for_the_uniform_distribution():
    """M1.6 gate: uniform entropy equals log(n) [MML p.606]."""
    n = 8
    uniform = np.full(n, 1.0 / n)
    assert _require(s1.entropy, uniform) == pytest.approx(np.log(n))
    spike = np.zeros(n)
    spike[0] = 1.0
    assert _require(s1.entropy, spike) == pytest.approx(0.0, abs=1e-12)


def test_gate_kl_is_zero_on_identical_distributions_and_never_negative():
    """M1.6 gate: KL(p, p) = 0 and KL >= 0 over 1000 pairs [MML p.606]."""
    rng = np.random.default_rng(5)
    p = rng.random(6)
    p = p / p.sum()
    assert _require(s1.kl_divergence, p, p) == pytest.approx(0.0, abs=1e-12)
    for _ in range(1000):
        a = rng.random(6)
        b = rng.random(6)
        a, b = a / a.sum(), b / b.sum()
        assert _require(s1.kl_divergence, a, b) >= -1e-12


def test_gate_mle_recovers_the_generating_parameters():
    """M1.6 gate: recovery within three standard errors [MML p.619]."""
    rng = np.random.default_rng(6)
    n, mu, sigma = 100_000, 1.75, 2.5
    samples = rng.normal(loc=mu, scale=sigma, size=n)
    mu_hat, var_hat = _require(s1.mle_gaussian, samples)
    standard_error = sigma / np.sqrt(n)
    assert abs(mu_hat - mu) < 3.0 * standard_error
    assert var_hat == pytest.approx(sigma**2, rel=0.02)
