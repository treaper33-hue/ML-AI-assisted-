"""Stage 1 exercises: the mathematical engine, built by you.

Every routine here already exists in NumPy. Writing it yourself is the
point, and the NumPy version is your test oracle -- which is why the gates
in `tests/test_stage1_gates.py` compare against it rather than against a
number you could talk yourself into.

Work in order. Each function names its anchor; read those pages first, as
the Analyst, and only then implement, as the Coder.

The gate tests skip while a function is unimplemented, so the suite stays
green for the rest of the testbed. A skipped gate is NOT a pass: the status
board in VERIFICATION.md stays at '--' until the test runs and passes.
"""

from typing import Callable, Tuple

import numpy as np


def lp_norm(x: np.ndarray, p: float = 2.0) -> float:
    """Computes the l_p norm of a vector.

    Anchor: [MML p.49] norms and distances.

    Args:
        x: A 1-D vector.
        p: The order; use `float('inf')` for the maximum norm.

    Returns:
        The norm as a float.
    """
    raise NotImplementedError('Stage 1, M1.1 -- read [MML p.49]')


def gram_schmidt(a: np.ndarray) -> np.ndarray:
    """Orthonormalises the columns of a matrix.

    Anchors: [MML p.67] the process, [MML p.89] the implementation.

    Gate: Q^T Q must equal the identity to 1e-10. Build a Hilbert matrix
    and watch the classical algorithm lose orthogonality -- that failure is
    a required observation, not an accident.

    Args:
        a: A matrix with linearly independent columns, shape (n, k).

    Returns:
        A matrix Q of shape (n, k) with orthonormal columns.
    """
    raise NotImplementedError('Stage 1, M1.1 -- read [MML p.67], [MML p.89]')


def lu_decompose(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes the LU decomposition with partial pivoting.

    Anchors: [MML p.168] the decomposition, [MML p.171] the implementation.

    Gate: ||PA - LU|| < 1e-10.

    Args:
        a: A square matrix.

    Returns:
        A tuple (P, L, U) with P a permutation matrix.
    """
    raise NotImplementedError('Stage 1, M1.2 -- read [MML p.171]')


def power_iteration(
    a: np.ndarray,
    iters: int = 1000,
    tol: float = 1e-12,
) -> Tuple[float, np.ndarray]:
    """Finds the dominant eigenpair of a symmetric matrix.

    Anchors: [MML p.189] eigenvalues, [MML p.217] computing them.

    Args:
        a: A symmetric matrix.
        iters: Maximum iterations.
        tol: Convergence tolerance on the eigenvector update.

    Returns:
        A tuple (eigenvalue, eigenvector) with the vector normalised.
    """
    raise NotImplementedError('Stage 1, M1.3 -- read [MML p.217]')


def svd_lowrank(a: np.ndarray, k: int) -> np.ndarray:
    """Returns the best rank-k approximation of a matrix.

    Anchor: [MML p.207] the singular value decomposition.

    Gate: the Frobenius error must equal the norm of the discarded
    singular values (Eckart-Young) to 1e-10. You may call `np.linalg.svd`
    here; the exercise is the truncation and the error identity.

    Args:
        a: The matrix to approximate.
        k: The target rank.

    Returns:
        The rank-k approximation, same shape as `a`.
    """
    raise NotImplementedError('Stage 1, M1.3 -- read [MML p.207]')


def gradient_descent(
    grad: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    lr: float,
    iters: int,
    momentum: float = 0.0,
) -> np.ndarray:
    """Minimises a function given its gradient.

    Anchors: [MML p.381] the one-dimensional case, [MML p.481] the full
    multivariable form.

    Gate: converges on a quadratic for lr < 2/L and diverges above it. The
    divergence is a required result -- record it, do not clamp it away.

    Args:
        grad: Gradient function.
        x0: Starting point.
        lr: Learning rate.
        iters: Number of steps.
        momentum: Heavy-ball coefficient; 0.0 gives plain descent.

    Returns:
        The final iterate.
    """
    raise NotImplementedError('Stage 1, M1.4 -- read [MML p.381], [MML p.481]')


def numeric_jacobian(
    f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    h: float = 1e-6,
) -> np.ndarray:
    """Approximates the Jacobian by central differences.

    Anchors: [MML p.448] partial derivatives, [MML p.460] vector-valued
    derivatives. This is backpropagation in miniature -- if the chain rule
    here is hazy, [PTG p.45] will stay magic.

    Args:
        f: A function from R^n to R^m.
        x: The point of evaluation, shape (n,).
        h: Step size.

    Returns:
        The Jacobian, shape (m, n).
    """
    raise NotImplementedError('Stage 1, M1.5 -- read [MML p.460]')


def entropy(p: np.ndarray, base: float = np.e) -> float:
    """Computes the Shannon entropy of a probability vector.

    Anchor: [MML p.606] information theory.

    Args:
        p: Probabilities summing to 1.
        base: Logarithm base; e gives nats, 2 gives bits.

    Returns:
        The entropy. Zero probabilities contribute zero.
    """
    raise NotImplementedError('Stage 1, M1.6 -- read [MML p.606]')


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Computes the Kullback-Leibler divergence from q to p.

    Anchor: [MML p.606]. This is the same quantity as the cross-entropy
    loss at [PTG p.38]; write that derivation out in your notes.

    Gate: KL(p, p) == 0 and KL >= 0 across 1000 random pairs.

    Args:
        p: Reference distribution.
        q: Comparison distribution.

    Returns:
        The divergence in nats.

    Raises:
        ValueError: If q is zero where p is positive.
    """
    raise NotImplementedError('Stage 1, M1.6 -- read [MML p.606]')


def mle_gaussian(samples: np.ndarray) -> Tuple[float, float]:
    """Maximum-likelihood estimate of a Gaussian's parameters.

    Anchor: [MML p.619] maximum likelihood estimation.

    Gate: recovers mu and sigma squared from 1e5 samples within three
    standard errors.

    Args:
        samples: Observed values.

    Returns:
        A tuple (mu_hat, sigma_squared_hat).
    """
    raise NotImplementedError('Stage 1, M1.6 -- read [MML p.619]')
