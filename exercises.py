"""Every training exercise from *Machine Learning: The Working Text*.

One file, one dependency (NumPy). PyTorch is used by Part 3 where it is
installed, and those exercises report SKIP rather than passing quietly when
it is not.

How it works:

  * Implement the stubs in the STUDENT IMPLEMENTATIONS section. Each one
    raises NotImplementedError until you replace it.
  * Run this file. Each exercise reports PASS, FAIL, TODO or SKIP against a
    numeric gate taken from the book.
  * TODO is never a pass. The summary says so explicitly.

Prefer a notebook? `python exercises.py --colab` prints how to work
these same exercises in Google Colab; the notebooks in `notebooks/` are
generated from this file, so the gates are identical.

Usage:
    python exercises.py                  run everything
    python exercises.py --part 1         run one part
    python exercises.py --exercise 5.2   run one exercise
    python exercises.py --list           show the index
    python exercises.py --solutions      run the reference implementations
    python exercises.py --verbose        show the gate detail for each

Exit code is 0 only when nothing failed and nothing was left TODO.
"""

import argparse
import dataclasses
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

NOTEBOOKS = {
    1: 'notebooks/part1_mathematical_engine.ipynb',
    2: 'notebooks/part2_learning_from_data.ipynb',
    3: 'notebooks/part3_neural_networks.ipynb',
    4: 'notebooks/part4_sequences_uncertainty.ipynb',
}

PASS = 'PASS'
FAIL = 'FAIL'
TODO = 'TODO'
SKIP = 'SKIP'


class NotImplementedYet(NotImplementedError):
    """Raised by a stub the reader has not filled in yet."""


# ===================================================================== #
#                      STUDENT IMPLEMENTATIONS                          #
#                                                                       #
#  Replace each `raise NotImplementedYet(...)` with your own code.      #
#  The chapter and page anchors tell you where to read first.           #
# ===================================================================== #


class Student:
    """Your implementations. Every method is a static function."""

    # ---------------- Part 1: the mathematical engine ---------------- #

    @staticmethod
    def lp_norm(x: np.ndarray, p: float) -> float:
        """Returns the l_p norm of a vector. [Ch 2 · MML p.49]"""
        raise NotImplementedYet('1.1 lp_norm')

    @staticmethod
    def gram_schmidt_classical(a: np.ndarray) -> np.ndarray:
        """Orthonormalises columns, classical version. [Ch 2 · MML p.67]"""
        raise NotImplementedYet('2.1 gram_schmidt_classical')

    @staticmethod
    def gram_schmidt_modified(a: np.ndarray) -> np.ndarray:
        """Orthonormalises columns, modified version. [Ch 2 · MML p.89]"""
        raise NotImplementedYet('2.2 gram_schmidt_modified')

    @staticmethod
    def projection_matrix(v: np.ndarray) -> np.ndarray:
        """Returns the matrix projecting onto span(v). [Ch 2]"""
        raise NotImplementedYet('2.3 projection_matrix')

    @staticmethod
    def lu_decompose(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                             np.ndarray]:
        """Returns (P, L, U) with PA = LU. [Ch 4 · MML p.171]"""
        raise NotImplementedYet('4.1 lu_decompose')

    @staticmethod
    def lu_solve(p: np.ndarray, l: np.ndarray, u: np.ndarray,
                 b: np.ndarray) -> np.ndarray:
        """Solves Ax = b from a PA = LU factorisation. [Ch 4]"""
        raise NotImplementedYet('4.2 lu_solve')

    @staticmethod
    def det_via_lu(a: np.ndarray) -> float:
        """Returns det(A) from its LU factorisation. [Ch 4 · MML p.182]"""
        raise NotImplementedYet('4.3 det_via_lu')

    @staticmethod
    def power_iteration(a: np.ndarray) -> Tuple[float, np.ndarray]:
        """Returns the dominant eigenpair. [Ch 5 · MML p.217]"""
        raise NotImplementedYet('5.1 power_iteration')

    @staticmethod
    def svd_lowrank(a: np.ndarray, k: int) -> np.ndarray:
        """Returns the best rank-k approximation. [Ch 5 · MML p.207]"""
        raise NotImplementedYet('5.2 svd_lowrank')

    @staticmethod
    def pca(x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (components, explained_variance_ratio). [Ch 5]"""
        raise NotImplementedYet('5.3 pca')

    @staticmethod
    def central_difference(f: Callable, x: float, h: float) -> float:
        """Approximates f'(x) by central difference. [Ch 6]"""
        raise NotImplementedYet('6.1 central_difference')

    @staticmethod
    def gradient_descent(grad: Callable, x0: np.ndarray, lr: float,
                         iters: int, momentum: float = 0.0) -> np.ndarray:
        """Minimises via gradient descent. [Ch 6 · MML p.381, p.481]"""
        raise NotImplementedYet('6.2 gradient_descent')

    @staticmethod
    def numeric_jacobian(f: Callable, x: np.ndarray,
                         h: float = 1e-6) -> np.ndarray:
        """Approximates the Jacobian of f at x. [Ch 7 · MML p.460]"""
        raise NotImplementedYet('7.1 numeric_jacobian')

    @staticmethod
    def two_layer_backward(w1: np.ndarray, b1: np.ndarray, w2: np.ndarray,
                           b2: np.ndarray, x: np.ndarray,
                           y: float) -> Tuple[float, Dict[str, np.ndarray]]:
        """Forward and backward pass of a two-layer tanh net. [Ch 7]"""
        raise NotImplementedYet('7.2 two_layer_backward')

    @staticmethod
    def shell_fraction(d: int, eps: float) -> float:
        """Fraction of a unit ball's volume in the outer shell. [Ch 7]"""
        raise NotImplementedYet('7.3 shell_fraction')

    @staticmethod
    def entropy(p: np.ndarray) -> float:
        """Shannon entropy in nats. [Ch 8 · MML p.606]"""
        raise NotImplementedYet('8.1 entropy')

    @staticmethod
    def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """KL divergence from q to p, in nats. [Ch 8 · MML p.606]"""
        raise NotImplementedYet('8.2 kl_divergence')

    @staticmethod
    def mle_gaussian(samples: np.ndarray) -> Tuple[float, float]:
        """Maximum-likelihood (mu, sigma^2). [Ch 8 · MML p.619]"""
        raise NotImplementedYet('8.3 mle_gaussian')

    @staticmethod
    def welford(values: np.ndarray) -> Tuple[float, float]:
        """One-pass stable (mean, sample variance). [Ch 8 · MML p.589]"""
        raise NotImplementedYet('8.4 welford')

    @staticmethod
    def bayes_posterior(sensitivity: float, specificity: float,
                        prevalence: float) -> float:
        """P(disease | positive test). [Ch 8 · MML p.515]"""
        raise NotImplementedYet('8.5 bayes_posterior')

    # ---------------- Part 2: learning from data --------------------- #

    @staticmethod
    def select_features(x_train: np.ndarray, y_train: np.ndarray,
                        k: int) -> np.ndarray:
        """Indices of the k features most correlated with the target.

        Must use the TRAINING rows only. [Ch 10 · TSP p.158]
        """
        raise NotImplementedYet('10.1 select_features')

    @staticmethod
    def assert_no_lookforward(folds: Sequence[Tuple[np.ndarray,
                                                    np.ndarray]]) -> None:
        """Raises AssertionError if a fold leaks the future. [Ch 10, 23]"""
        raise NotImplementedYet('10.2 assert_no_lookforward')

    @staticmethod
    def standardize_fit(x_train: np.ndarray) -> Tuple[np.ndarray,
                                                      np.ndarray]:
        """Returns (mean, std) computed on the training rows. [Ch 11]"""
        raise NotImplementedYet('11.1 standardize_fit')

    @staticmethod
    def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
        """Ridge coefficients, intercept last, unpenalised. [Ch 12]"""
        raise NotImplementedYet('12.1 ridge_fit')

    @staticmethod
    def logistic_gradient(w: np.ndarray, x: np.ndarray,
                          y: np.ndarray) -> np.ndarray:
        """Gradient of mean binary cross-entropy w.r.t. w. [Ch 12]"""
        raise NotImplementedYet('12.2 logistic_gradient')

    @staticmethod
    def gini_impurity(labels: np.ndarray) -> float:
        """Gini impurity of a label vector. [Ch 13]"""
        raise NotImplementedYet('13.1 gini_impurity')

    @staticmethod
    def precision_recall_f1(y_true: np.ndarray,
                            y_pred: np.ndarray) -> Tuple[float, float,
                                                         float]:
        """Binary precision, recall and F1 for the positive class. [Ch 14]"""
        raise NotImplementedYet('14.1 precision_recall_f1')

    @staticmethod
    def bootstrap_ci(metric_fn: Callable, y_true: np.ndarray,
                     y_pred: np.ndarray, n_boot: int, level: float,
                     seed: int) -> Tuple[float, float]:
        """Percentile bootstrap confidence interval. [Ch 14]"""
        raise NotImplementedYet('14.2 bootstrap_ci')

    @staticmethod
    def kmeans(x: np.ndarray, k: int, iters: int,
               seed: int) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """Lloyd's algorithm: (labels, centroids, objective history). [Ch 15]"""
        raise NotImplementedYet('15.1 kmeans')

    # ---------------- Part 3: neural networks ------------------------ #

    @staticmethod
    def stable_softmax(z: np.ndarray) -> np.ndarray:
        """Softmax over the last axis, overflow-safe. [Ch 17 · PTG p.37]"""
        raise NotImplementedYet('17.1 stable_softmax')

    @staticmethod
    def cross_entropy_grad(logits: np.ndarray,
                           one_hot: np.ndarray) -> np.ndarray:
        """Gradient of cross-entropy w.r.t. the logits. [Ch 17 · PTG p.40]"""
        raise NotImplementedYet('17.2 cross_entropy_grad')

    @staticmethod
    def conv2d(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Valid single-channel 2-D cross-correlation. [Ch 19 · PTG p.128]"""
        raise NotImplementedYet('19.1 conv2d')

    @staticmethod
    def attention(q: np.ndarray, k: np.ndarray,
                  v: np.ndarray) -> np.ndarray:
        """Scaled dot-product attention. [Ch 21 · PTG p.320]"""
        raise NotImplementedYet('21.1 attention')

    @staticmethod
    def torch_training_step(model, xb, yb, optimiser, loss_fn) -> float:
        """One zero_grad / forward / backward / step. [Ch 18 · PTG p.62]"""
        raise NotImplementedYet('18.1 torch_training_step')

    # ---------------- Part 4: sequences and uncertainty -------------- #

    @staticmethod
    def acf(series: np.ndarray, max_lag: int) -> np.ndarray:
        """Sample autocorrelation, lags 0..max_lag. [Ch 22 · TSP p.84]"""
        raise NotImplementedYet('22.1 acf')

    @staticmethod
    def additive_decompose(series: np.ndarray,
                           period: int) -> Tuple[np.ndarray, np.ndarray,
                                                 np.ndarray]:
        """Returns (trend, seasonal, residual). [Ch 22 · TSP p.66]"""
        raise NotImplementedYet('22.2 additive_decompose')

    @staticmethod
    def rolling_window_cv(n: int, n_folds: int, horizon: int,
                          window: int) -> List[Tuple[np.ndarray,
                                                     np.ndarray]]:
        """Rolling-origin folds. [Ch 23 · TSP p.134]"""
        raise NotImplementedYet('23.1 rolling_window_cv')

    @staticmethod
    def naive_scale(y_train: np.ndarray, seasonality: int) -> float:
        """MAE of the seasonal naive forecast on TRAIN. [Ch 23 · TSP p.150]"""
        raise NotImplementedYet('23.2 naive_scale')

    @staticmethod
    def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray,
             seasonality: int) -> float:
        """Mean absolute scaled error. [Ch 23 · TSP p.150]"""
        raise NotImplementedYet('23.3 mase')

    @staticmethod
    def absolute_percentage_error(y_true: np.ndarray,
                                  y_pred: np.ndarray) -> np.ndarray:
        """Element-wise APE. Must RAISE at zero targets. [TSP p.144]"""
        raise NotImplementedYet('23.4 absolute_percentage_error')

    @staticmethod
    def split_conformal(residuals: np.ndarray, alpha: float) -> float:
        """Conformal half-width for 1-alpha coverage. [Ch 25 · TSP p.325]"""
        raise NotImplementedYet('25.1 split_conformal')

    # -------- exercises from the worked examples and derivations ------ #

    @staticmethod
    def eigenvalues_2x2(a: np.ndarray) -> Tuple[float, float]:
        """Eigenvalues of a 2x2 via the characteristic polynomial. [Ch 5]"""
        raise NotImplementedYet('5.4 eigenvalues_2x2')

    @staticmethod
    def singular_values(a: np.ndarray) -> np.ndarray:
        """Singular values as sqrt of the eigenvalues of AᵀA. [Ch 5]"""
        raise NotImplementedYet('5.5 singular_values')

    @staticmethod
    def optimal_threshold(cost_fp: float, cost_fn: float) -> float:
        """Decision threshold from the two error costs. [Ch 9]"""
        raise NotImplementedYet('9.1 optimal_threshold')

    @staticmethod
    def soft_threshold(c: float, alpha: float) -> float:
        """The 1-D lasso solution: sign(c)·max(|c| - alpha, 0). [Ch 12]"""
        raise NotImplementedYet('12.3 soft_threshold')

    @staticmethod
    def gini_gain(parent: np.ndarray, left: np.ndarray,
                  right: np.ndarray) -> float:
        """Weighted Gini impurity reduction of a split. [Ch 13]"""
        raise NotImplementedYet('13.2 gini_gain')

    @staticmethod
    def conv_output_size(in_size: int, kernel: int, stride: int,
                         padding: int, dilation: int) -> int:
        """Convolution output size. [Ch 19 · PTG p.129]"""
        raise NotImplementedYet('19.2 conv_output_size')

    @staticmethod
    def ar1_acf(phi: float, max_lag: int) -> np.ndarray:
        """Theoretical ACF of an AR(1): rho(k) = phi^k. [Ch 22]"""
        raise NotImplementedYet('22.3 ar1_acf')

    @staticmethod
    def winkler_score(y: float, lower: float, upper: float,
                      alpha: float) -> float:
        """Interval score: width plus an out-of-interval penalty. [Ch 25]"""
        raise NotImplementedYet('25.2 winkler_score')


# ===================================================================== #
#                      REFERENCE IMPLEMENTATIONS                        #
#                                                                       #
#  Run with --solutions to confirm every gate is satisfiable. Read      #
#  these only after your own attempt: the struggle is the lesson.       #
# ===================================================================== #


class Solutions:
    """Reference implementations of every exercise."""

    @staticmethod
    def lp_norm(x: np.ndarray, p: float) -> float:
        """Returns the l_p norm of a vector."""
        x = np.asarray(x, dtype=np.float64)
        if np.isinf(p):
            return float(np.max(np.abs(x)))
        return float(np.sum(np.abs(x) ** p) ** (1.0 / p))

    @staticmethod
    def gram_schmidt_classical(a: np.ndarray) -> np.ndarray:
        """Orthonormalises columns, projecting against the original column."""
        a = np.asarray(a, dtype=np.float64)
        n, k = a.shape
        q = np.zeros((n, k))
        for j in range(k):
            v = a[:, j].copy()
            for i in range(j):
                v = v - (q[:, i] @ a[:, j]) * q[:, i]
            norm = np.linalg.norm(v)
            if norm < 1e-300:
                raise ValueError(f'column {j} is dependent')
            q[:, j] = v / norm
        return q

    @staticmethod
    def gram_schmidt_modified(a: np.ndarray) -> np.ndarray:
        """Orthonormalises columns, projecting against the running residual."""
        a = np.asarray(a, dtype=np.float64)
        n, k = a.shape
        q = np.zeros((n, k))
        for j in range(k):
            v = a[:, j].copy()
            for i in range(j):
                v = v - (q[:, i] @ v) * q[:, i]
            norm = np.linalg.norm(v)
            if norm < 1e-300:
                raise ValueError(f'column {j} is dependent')
            q[:, j] = v / norm
        return q

    @staticmethod
    def projection_matrix(v: np.ndarray) -> np.ndarray:
        """Returns the matrix projecting onto span(v)."""
        v = np.asarray(v, dtype=np.float64).reshape(-1, 1)
        return (v @ v.T) / float((v.T @ v).item())

    @staticmethod
    def lu_decompose(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                             np.ndarray]:
        """Factors a square matrix as PA = LU with partial pivoting."""
        a = np.asarray(a, dtype=np.float64)
        n = a.shape[0]
        u = a.copy()
        l = np.eye(n)
        p = np.eye(n)
        for k in range(n - 1):
            pivot = int(np.argmax(np.abs(u[k:, k]))) + k
            if abs(u[pivot, k]) < 1e-14:
                raise ValueError('singular to working precision')
            if pivot != k:
                u[[k, pivot], k:] = u[[pivot, k], k:]
                l[[k, pivot], :k] = l[[pivot, k], :k]
                p[[k, pivot], :] = p[[pivot, k], :]
            for row in range(k + 1, n):
                factor = u[row, k] / u[k, k]
                l[row, k] = factor
                u[row, k:] -= factor * u[k, k:]
        return p, l, u

    @staticmethod
    def lu_solve(p: np.ndarray, l: np.ndarray, u: np.ndarray,
                 b: np.ndarray) -> np.ndarray:
        """Solves Ax = b given PA = LU."""
        pb = p @ np.asarray(b, dtype=np.float64)
        n = len(pb)
        y = np.zeros(n)
        for i in range(n):
            y[i] = pb[i] - l[i, :i] @ y[:i]
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            x[i] = (y[i] - u[i, i + 1:] @ x[i + 1:]) / u[i, i]
        return x

    @staticmethod
    def det_via_lu(a: np.ndarray) -> float:
        """Returns det(A) from its LU factorisation."""
        p, _, u = Solutions.lu_decompose(a)
        swaps = int(round(np.sum(np.abs(p - np.eye(p.shape[0]))) / 2.0))
        return float(((-1.0) ** swaps) * np.prod(np.diag(u)))

    @staticmethod
    def power_iteration(a: np.ndarray) -> Tuple[float, np.ndarray]:
        """Returns the dominant eigenpair of a symmetric matrix."""
        a = np.asarray(a, dtype=np.float64)
        rng = np.random.default_rng(0)
        v = rng.normal(size=a.shape[0])
        v /= np.linalg.norm(v)
        for _ in range(10_000):
            w = a @ v
            norm = np.linalg.norm(w)
            if norm < 1e-300:
                return 0.0, v
            w /= norm
            if min(np.linalg.norm(w - v), np.linalg.norm(w + v)) < 1e-14:
                v = w
                break
            v = w
        return float(v @ a @ v), v

    @staticmethod
    def svd_lowrank(a: np.ndarray, k: int) -> np.ndarray:
        """Returns the best rank-k approximation."""
        u, s, vt = np.linalg.svd(np.asarray(a, dtype=np.float64),
                                 full_matrices=False)
        return (u[:, :k] * s[:k]) @ vt[:k, :]

    @staticmethod
    def pca(x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (components, explained_variance_ratio)."""
        x = np.asarray(x, dtype=np.float64)
        centred = x - x.mean(axis=0)
        _, s, vt = np.linalg.svd(centred, full_matrices=False)
        variance = s ** 2 / (x.shape[0] - 1)
        return vt[:k], variance[:k] / variance.sum()

    @staticmethod
    def central_difference(f: Callable, x: float, h: float) -> float:
        """Approximates f'(x) by central difference."""
        return float((f(x + h) - f(x - h)) / (2.0 * h))

    @staticmethod
    def gradient_descent(grad: Callable, x0: np.ndarray, lr: float,
                         iters: int, momentum: float = 0.0) -> np.ndarray:
        """Minimises a function given its gradient."""
        x = np.asarray(x0, dtype=np.float64).copy()
        velocity = np.zeros_like(x)
        for _ in range(iters):
            g = np.asarray(grad(x), dtype=np.float64)
            velocity = momentum * velocity - lr * g
            x = x + velocity
            if not np.all(np.isfinite(x)):
                break
        return x

    @staticmethod
    def numeric_jacobian(f: Callable, x: np.ndarray,
                         h: float = 1e-6) -> np.ndarray:
        """Approximates the Jacobian of f at x."""
        x = np.asarray(x, dtype=np.float64)
        base = np.atleast_1d(np.asarray(f(x), dtype=np.float64))
        jac = np.zeros((base.size, x.size))
        for i in range(x.size):
            step = np.zeros_like(x)
            step[i] = h
            plus = np.atleast_1d(np.asarray(f(x + step), dtype=np.float64))
            minus = np.atleast_1d(np.asarray(f(x - step), dtype=np.float64))
            jac[:, i] = (plus - minus) / (2.0 * h)
        return jac

    @staticmethod
    def two_layer_backward(w1: np.ndarray, b1: np.ndarray, w2: np.ndarray,
                           b2: np.ndarray, x: np.ndarray,
                           y: float) -> Tuple[float, Dict[str, np.ndarray]]:
        """Forward and backward pass of a two-layer tanh network."""
        z1 = w1 @ x + b1
        a1 = np.tanh(z1)
        z2 = w2 @ a1 + b2
        loss = 0.5 * float(((z2 - y) ** 2).item())
        d2 = (z2 - y)
        grads = {'w2': np.outer(d2, a1), 'b2': d2}
        d1 = (w2.T @ d2) * (1.0 - a1 ** 2)
        grads['w1'] = np.outer(d1, x)
        grads['b1'] = d1
        return loss, grads

    @staticmethod
    def shell_fraction(d: int, eps: float) -> float:
        """Fraction of a unit ball's volume within eps of the surface."""
        return float(1.0 - (1.0 - eps) ** d)

    @staticmethod
    def entropy(p: np.ndarray) -> float:
        """Shannon entropy in nats."""
        p = np.asarray(p, dtype=np.float64)
        nonzero = p[p > 0]
        return float(-np.sum(nonzero * np.log(nonzero)))

    @staticmethod
    def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """KL divergence from q to p, in nats."""
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        mask = p > 0
        if np.any(q[mask] <= 0):
            raise ValueError('KL is infinite where q = 0 and p > 0')
        return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))

    @staticmethod
    def mle_gaussian(samples: np.ndarray) -> Tuple[float, float]:
        """Maximum-likelihood mean and variance."""
        x = np.asarray(samples, dtype=np.float64)
        mu = float(x.mean())
        return mu, float(((x - mu) ** 2).mean())

    @staticmethod
    def welford(values: np.ndarray) -> Tuple[float, float]:
        """One-pass stable mean and sample variance."""
        count, mean, m2 = 0, 0.0, 0.0
        for value in np.asarray(values, dtype=np.float64).ravel():
            count += 1
            delta = value - mean
            mean += delta / count
            m2 += delta * (value - mean)
        return mean, (m2 / (count - 1) if count > 1 else 0.0)

    @staticmethod
    def bayes_posterior(sensitivity: float, specificity: float,
                        prevalence: float) -> float:
        """P(disease | positive test)."""
        true_pos = sensitivity * prevalence
        false_pos = (1.0 - specificity) * (1.0 - prevalence)
        return float(true_pos / (true_pos + false_pos))

    @staticmethod
    def select_features(x_train: np.ndarray, y_train: np.ndarray,
                        k: int) -> np.ndarray:
        """Indices of the k training-correlated features."""
        centred = y_train - y_train.mean()
        scores = np.abs(x_train.T @ centred)
        return np.argsort(scores)[-k:]

    @staticmethod
    def assert_no_lookforward(folds) -> None:
        """Raises AssertionError if a fold leaks the future."""
        for i, (train_idx, test_idx) in enumerate(folds):
            if len(train_idx) == 0 or len(test_idx) == 0:
                raise AssertionError(f'fold {i}: empty split')
            if int(np.max(train_idx)) >= int(np.min(test_idx)):
                raise AssertionError(f'fold {i}: lookforward leak')

    @staticmethod
    def standardize_fit(x_train: np.ndarray) -> Tuple[np.ndarray,
                                                      np.ndarray]:
        """Returns (mean, std) from the training rows only."""
        x_train = np.asarray(x_train, dtype=np.float64)
        std = x_train.std(axis=0)
        std[std == 0] = 1.0
        return x_train.mean(axis=0), std

    @staticmethod
    def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
        """Ridge coefficients with an unpenalised intercept, placed last."""
        if alpha < 0:
            raise ValueError('alpha must be non-negative')
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        x_mean, y_mean = x.mean(axis=0), y.mean()
        xc, yc = x - x_mean, y - y_mean
        d = x.shape[1]
        w = np.linalg.solve(xc.T @ xc + alpha * np.eye(d), xc.T @ yc)
        return np.append(w, y_mean - x_mean @ w)

    @staticmethod
    def logistic_gradient(w: np.ndarray, x: np.ndarray,
                          y: np.ndarray) -> np.ndarray:
        """Gradient of mean binary cross-entropy with respect to w."""
        z = x @ w
        p = 1.0 / (1.0 + np.exp(-z))
        return x.T @ (p - y) / len(y)

    @staticmethod
    def gini_impurity(labels: np.ndarray) -> float:
        """Gini impurity of a label vector."""
        labels = np.asarray(labels)
        if labels.size == 0:
            return 0.0
        _, counts = np.unique(labels, return_counts=True)
        p = counts / labels.size
        return float(1.0 - np.sum(p ** 2))

    @staticmethod
    def precision_recall_f1(y_true: np.ndarray,
                            y_pred: np.ndarray) -> Tuple[float, float,
                                                         float]:
        """Binary precision, recall and F1 for the positive class."""
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        denom = precision + recall
        f1 = 2 * precision * recall / denom if denom else 0.0
        return precision, recall, f1

    @staticmethod
    def bootstrap_ci(metric_fn: Callable, y_true: np.ndarray,
                     y_pred: np.ndarray, n_boot: int, level: float,
                     seed: int) -> Tuple[float, float]:
        """Percentile bootstrap confidence interval."""
        rng = np.random.default_rng(seed)
        n = len(y_true)
        samples = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, n, size=n)
            samples[i] = metric_fn(y_true[idx], y_pred[idx])
        tail = (1.0 - level) / 2.0
        return (float(np.quantile(samples, tail)),
                float(np.quantile(samples, 1.0 - tail)))

    @staticmethod
    def kmeans(x: np.ndarray, k: int, iters: int,
               seed: int) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """Lloyd's algorithm with a fixed seed."""
        x = np.asarray(x, dtype=np.float64)
        rng = np.random.default_rng(seed)
        centroids = x[rng.choice(len(x), size=k, replace=False)].copy()
        history: List[float] = []
        labels = np.zeros(len(x), dtype=int)
        for _ in range(iters):
            distances = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
            labels = np.argmin(distances, axis=1)
            history.append(float(distances[np.arange(len(x)), labels].sum()))
            for j in range(k):
                members = x[labels == j]
                if len(members):
                    centroids[j] = members.mean(axis=0)
        return labels, centroids, history

    @staticmethod
    def stable_softmax(z: np.ndarray) -> np.ndarray:
        """Softmax over the last axis, overflow-safe."""
        z = np.asarray(z, dtype=np.float64)
        shifted = z - z.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=-1, keepdims=True)

    @staticmethod
    def cross_entropy_grad(logits: np.ndarray,
                           one_hot: np.ndarray) -> np.ndarray:
        """Gradient of cross-entropy with respect to the logits."""
        return Solutions.stable_softmax(logits) - one_hot

    @staticmethod
    def conv2d(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Valid single-channel 2-D cross-correlation."""
        x = np.asarray(x, dtype=np.float64)
        kernel = np.asarray(kernel, dtype=np.float64)
        kh, kw = kernel.shape
        out_h, out_w = x.shape[0] - kh + 1, x.shape[1] - kw + 1
        patches = np.lib.stride_tricks.sliding_window_view(x, (kh, kw))
        cols = patches.reshape(out_h * out_w, kh * kw)
        return (cols @ kernel.ravel()).reshape(out_h, out_w)

    @staticmethod
    def attention(q: np.ndarray, k: np.ndarray,
                  v: np.ndarray) -> np.ndarray:
        """Scaled dot-product attention."""
        scores = (q @ k.T) / np.sqrt(k.shape[-1])
        return Solutions.stable_softmax(scores) @ v

    @staticmethod
    def torch_training_step(model, xb, yb, optimiser, loss_fn) -> float:
        """One zero_grad / forward / backward / step."""
        optimiser.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimiser.step()
        # .detach() first: converting a tensor that still carries
        # requires_grad straight to a float warns, and in a loop it
        # quietly keeps the graph alive.
        return float(loss.detach())

    @staticmethod
    def acf(series: np.ndarray, max_lag: int) -> np.ndarray:
        """Sample autocorrelation for lags 0..max_lag."""
        y = np.asarray(series, dtype=np.float64)
        y = y - y.mean()
        denom = float(y @ y)
        return np.array([float(y[k:] @ y[:len(y) - k]) / denom
                         for k in range(max_lag + 1)])

    @staticmethod
    def additive_decompose(series: np.ndarray,
                           period: int) -> Tuple[np.ndarray, np.ndarray,
                                                 np.ndarray]:
        """Centred-moving-average additive decomposition."""
        y = np.asarray(series, dtype=np.float64)
        kernel = np.ones(period) / period
        trend = np.convolve(y, kernel, mode='same')
        half = period // 2
        trend[:half] = trend[half]
        trend[len(y) - half:] = trend[len(y) - half - 1]
        detrended = y - trend
        seasonal_means = np.array([detrended[i::period].mean()
                                   for i in range(period)])
        seasonal_means -= seasonal_means.mean()
        seasonal = np.array([seasonal_means[i % period]
                             for i in range(len(y))])
        return trend, seasonal, y - trend - seasonal

    @staticmethod
    def rolling_window_cv(n: int, n_folds: int, horizon: int,
                          window: int) -> List[Tuple[np.ndarray,
                                                     np.ndarray]]:
        """Rolling-origin folds with a constant training length."""
        needed = window + n_folds * horizon
        if needed > n:
            raise ValueError(f'need {needed} points, have {n}')
        folds = []
        for i in range(n_folds):
            end = window + i * horizon
            folds.append((np.arange(end - window, end),
                          np.arange(end, end + horizon)))
        return folds

    @staticmethod
    def naive_scale(y_train: np.ndarray, seasonality: int) -> float:
        """MAE of the seasonal naive forecast on the training series."""
        y = np.asarray(y_train, dtype=np.float64)
        if y.size <= seasonality:
            raise ValueError('training series too short')
        scale = float(np.mean(np.abs(y[seasonality:] - y[:-seasonality])))
        if scale == 0.0:
            raise ValueError('constant training series; MASE undefined')
        return scale

    @staticmethod
    def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray,
             seasonality: int) -> float:
        """Mean absolute scaled error."""
        mae = float(np.mean(np.abs(np.asarray(y_true, dtype=np.float64) -
                                   np.asarray(y_pred, dtype=np.float64))))
        return mae / Solutions.naive_scale(y_train, seasonality)

    @staticmethod
    def absolute_percentage_error(y_true: np.ndarray,
                                  y_pred: np.ndarray) -> np.ndarray:
        """Element-wise APE, undefined at zero targets."""
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        if np.any(y_true == 0.0):
            raise ValueError('APE is undefined for zero targets [TSP p.144]')
        return np.abs((y_true - y_pred) / y_true)

    @staticmethod
    def split_conformal(residuals: np.ndarray, alpha: float) -> float:
        """Conformal half-width for 1 - alpha coverage."""
        s = np.sort(np.asarray(residuals, dtype=np.float64))
        n = s.size
        k = int(np.ceil((n + 1) * (1.0 - alpha))) - 1
        return float(s[min(max(k, 0), n - 1)])

    @staticmethod
    def eigenvalues_2x2(a: np.ndarray) -> Tuple[float, float]:
        """Eigenvalues of a 2x2 from the characteristic polynomial."""
        a = np.asarray(a, dtype=np.float64)
        trace = float(a[0, 0] + a[1, 1])
        det = float(a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0])
        disc = trace ** 2 - 4.0 * det
        if disc < 0:
            raise ValueError('complex eigenvalues; this exercise is real-only')
        root = float(np.sqrt(disc))
        return (trace + root) / 2.0, (trace - root) / 2.0

    @staticmethod
    def singular_values(a: np.ndarray) -> np.ndarray:
        """Singular values as the square roots of the eigenvalues of AᵀA."""
        a = np.asarray(a, dtype=np.float64)
        eigenvalues = np.linalg.eigvalsh(a.T @ a)
        clipped = np.clip(eigenvalues, 0.0, None)
        return np.sqrt(clipped)[::-1]

    @staticmethod
    def optimal_threshold(cost_fp: float, cost_fn: float) -> float:
        """Decision threshold balancing the two expected costs."""
        if cost_fp < 0 or cost_fn < 0 or (cost_fp + cost_fn) == 0:
            raise ValueError('costs must be non-negative and not both zero')
        return float(cost_fp / (cost_fp + cost_fn))

    @staticmethod
    def soft_threshold(c: float, alpha: float) -> float:
        """The 1-D lasso solution."""
        return float(np.sign(c) * max(abs(c) - alpha, 0.0))

    @staticmethod
    def gini_gain(parent: np.ndarray, left: np.ndarray,
                  right: np.ndarray) -> float:
        """Weighted Gini impurity reduction of a split."""
        n = len(parent)
        if n == 0 or len(left) + len(right) != n:
            raise ValueError('children must partition the parent')
        weighted = (len(left) / n) * Solutions.gini_impurity(left) + \
                   (len(right) / n) * Solutions.gini_impurity(right)
        return float(Solutions.gini_impurity(parent) - weighted)

    @staticmethod
    def conv_output_size(in_size: int, kernel: int, stride: int,
                         padding: int, dilation: int) -> int:
        """Convolution output size along one axis."""
        effective = dilation * (kernel - 1)
        return int((in_size + 2 * padding - effective - 1) // stride) + 1

    @staticmethod
    def ar1_acf(phi: float, max_lag: int) -> np.ndarray:
        """Theoretical ACF of an AR(1)."""
        if abs(phi) >= 1.0:
            raise ValueError('|phi| must be < 1 for a stationary AR(1)')
        return np.array([phi ** k for k in range(max_lag + 1)])

    @staticmethod
    def winkler_score(y: float, lower: float, upper: float,
                      alpha: float) -> float:
        """Interval score: width plus an out-of-interval penalty."""
        width = upper - lower
        if y < lower:
            return float(width + (2.0 / alpha) * (lower - y))
        if y > upper:
            return float(width + (2.0 / alpha) * (y - upper))
        return float(width)


# ===================================================================== #
#                              THE GATES                                #
# ===================================================================== #


@dataclasses.dataclass
class Exercise:
    """One exercise and the numeric gate that decides it.

    Attributes:
        key: Identifier such as '5.2'.
        part: Which part of the book it belongs to.
        title: Short description.
        anchor: Where to read first.
        check: Callable taking the implementation namespace; raises on
            failure and may return a short detail string.
        needs_torch: Whether PyTorch must be installed.
    """

    key: str
    part: int
    title: str
    anchor: str
    check: Callable[[type], Optional[str]]
    needs_torch: bool = False


def _rng(seed: int = 0) -> np.random.Generator:
    """Returns a seeded generator so every gate is reproducible."""
    return np.random.default_rng(seed)


# ---- Part 1 gates ---------------------------------------------------- #

def _g_norm(impl) -> str:
    x = _rng(0).normal(size=64)
    for p in (1.0, 2.0, float('inf')):
        got = impl.lp_norm(x, p)
        want = float(np.linalg.norm(x, ord=p))
        assert abs(got - want) < 1e-10 * max(1.0, abs(want)), f'p={p}'
    assert abs(impl.lp_norm(np.array([3.0, 4.0]), 2.0) - 5.0) < 1e-12
    return 'matches numpy for p = 1, 2, inf'


def _g_vectorisation(impl) -> str:
    n = 200_000
    a, b = _rng(1).normal(size=n), _rng(2).normal(size=n)
    start = time.perf_counter()
    slow = sum(x * y for x, y in zip(a.tolist(), b.tolist()))
    py_s = time.perf_counter() - start
    start = time.perf_counter()
    fast = float(np.dot(a, b))
    np_s = max(time.perf_counter() - start, 1e-9)
    assert abs(slow - fast) < 1e-6 * max(1.0, abs(fast))
    return f'speedup on this machine: {py_s / np_s:.0f}x'


def _g_rank(impl) -> str:
    x = _rng(3).normal(size=(50, 5))
    x[:, 4] = x[:, 0] + 2.0 * x[:, 1]          # deliberately collinear
    rank = int(np.linalg.matrix_rank(x))
    assert rank == 4, f'expected rank 4, got {rank}'
    return 'collinear column detected (rank 4 of 5)'


def _g_gram_schmidt(impl) -> str:
    a = _rng(4).normal(size=(50, 6))
    q = impl.gram_schmidt_modified(a)
    err = float(np.abs(q.T @ q - np.eye(6)).max())
    assert err < 1e-10, f'QtQ deviates by {err:.2e}'
    return f'QtQ = I to {err:.1e}'


def _g_gram_schmidt_stability(impl) -> str:
    n = 10
    hilbert = np.array([[1.0 / (i + j + 1) for j in range(n)]
                        for i in range(n)])
    classical = float(np.abs(impl.gram_schmidt_classical(hilbert).T @
                             impl.gram_schmidt_classical(hilbert) -
                             np.eye(n)).max())
    modified = float(np.abs(impl.gram_schmidt_modified(hilbert).T @
                            impl.gram_schmidt_modified(hilbert) -
                            np.eye(n)).max())
    assert modified < classical, (
        f'modified ({modified:.2e}) should beat classical ({classical:.2e})')
    return f'classical {classical:.1e} vs modified {modified:.1e}'


def _g_projection(impl) -> str:
    v = _rng(5).normal(size=7)
    p = impl.projection_matrix(v)
    assert np.allclose(p @ p, p, atol=1e-12), 'P² != P'
    assert np.allclose(p, p.T, atol=1e-12), 'P != Pᵀ'
    return 'idempotent and symmetric to 1e-12'


def _g_rank_nullity(impl) -> str:
    rng = _rng(6)
    for _ in range(50):
        a = rng.normal(size=(8, 6)) @ rng.normal(size=(6, 6))
        rank = int(np.linalg.matrix_rank(a))
        nullity = a.shape[1] - rank
        assert rank + nullity == a.shape[1]
    return 'rank + nullity = n over 50 matrices'


def _g_rotation(impl) -> str:
    theta = 0.7
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    x = _rng(7).normal(size=2)
    assert abs(np.linalg.norm(rot @ x) - np.linalg.norm(x)) < 1e-12
    assert abs(np.linalg.det(rot) - 1.0) < 1e-12
    return 'norm preserved, determinant 1'


def _g_lu(impl) -> str:
    a = _rng(8).normal(size=(8, 8))
    p, l, u = impl.lu_decompose(a)
    err = float(np.abs(p @ a - l @ u).max())
    assert err < 1e-10, f'||PA - LU|| = {err:.2e}'
    assert np.allclose(np.tril(l), l, atol=1e-12), 'L is not lower-triangular'
    assert np.allclose(np.triu(u), u, atol=1e-12), 'U is not upper-triangular'
    return f'PA = LU to {err:.1e}'


def _g_lu_solve(impl) -> str:
    rng = _rng(9)
    a = rng.normal(size=(10, 10))
    b = rng.normal(size=10)
    p, l, u = impl.lu_decompose(a)
    mine = impl.lu_solve(p, l, u, b)
    want = np.linalg.solve(a, b)
    err = float(np.abs(mine - want).max())
    assert err < 1e-9, f'solution differs by {err:.2e}'
    return f'matches np.linalg.solve to {err:.1e}'


def _g_det(impl) -> str:
    a = _rng(10).normal(size=(7, 7))
    mine, want = impl.det_via_lu(a), float(np.linalg.det(a))
    err = abs(mine - want) / max(1.0, abs(want))
    assert err < 1e-9, f'relative error {err:.2e}'
    return f'matches np.linalg.det to {err:.1e} relative'


def _g_conditioning(impl) -> str:
    rng = _rng(11)
    n = 120
    u_mat, _, vt = np.linalg.svd(rng.normal(size=(n, n)))
    a = (u_mat * np.logspace(0, -10, n)) @ vt
    x_true = rng.normal(size=n)
    b = a @ x_true
    err_inv = float(np.linalg.norm(np.linalg.inv(a) @ b - x_true))
    err_solve = float(np.linalg.norm(np.linalg.solve(a, b) - x_true))
    assert err_solve <= err_inv, (
        f'solve {err_solve:.2e} should not exceed inv {err_inv:.2e}')
    return f'kappa {np.linalg.cond(a):.1e}; solve {err_solve:.1e} <= inv {err_inv:.1e}'


def _g_power_iteration(impl) -> str:
    base = _rng(12).normal(size=(20, 20))
    a = base @ base.T
    value, vector = impl.power_iteration(a)
    want = float(np.linalg.eigvalsh(a).max())
    assert abs(value - want) < 1e-6 * want, f'{value:.6f} vs {want:.6f}'
    residual = float(np.linalg.norm(a @ vector - value * vector))
    assert residual < 1e-6 * want, f'residual {residual:.2e}'
    return f'dominant eigenvalue to {abs(value - want) / want:.1e} relative'


def _g_eckart_young(impl) -> str:
    a = _rng(13).normal(size=(40, 15))
    s = np.linalg.svd(a, compute_uv=False)
    details = []
    for k in (1, 5, 10):
        predicted = float(np.sqrt(np.sum(s[k:] ** 2)))
        actual = float(np.linalg.norm(a - impl.svd_lowrank(a, k)))
        assert abs(predicted - actual) < 1e-9, f'k={k}'
        details.append(f'k={k}:{actual:.3f}')
    return 'Eckart-Young exact at ' + ', '.join(details)


def _g_pca(impl) -> str:
    rng = _rng(14)
    latent = rng.normal(size=(300, 2))
    mixing = rng.normal(size=(2, 5))
    x = latent @ mixing + 0.05 * rng.normal(size=(300, 5))
    components, ratio = impl.pca(x, 2)
    centred = x - x.mean(axis=0)
    cov = centred.T @ centred / (len(x) - 1)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    want = eigvals[:2] / eigvals.sum()
    assert np.allclose(ratio, want, atol=1e-8), 'variance ratios differ'
    assert abs(float(np.abs(components @ components.T)[0, 1])) < 1e-8
    return f'top-2 variance {ratio.sum():.3f}, components orthogonal'


def _g_central_difference(impl) -> str:
    exact = float(np.cos(1.0))
    errors = [abs(impl.central_difference(np.sin, 1.0, 10.0 ** -k) - exact)
              for k in range(1, 15)]
    best = int(np.argmin(errors))
    assert best not in (0, len(errors) - 1), (
        'the error curve should bottom out in the middle, not at an end')
    assert errors[best] < 1e-9
    return f'minimum error {errors[best]:.1e} at h = 1e-{best + 1}'


def _g_gd_converges(impl) -> str:
    curvature = np.array([4.0, 1.0])
    limit = 2.0 / (2.0 * curvature.max())
    grad = lambda x: 2.0 * curvature * x
    final = impl.gradient_descent(grad, np.array([5.0, -3.0]),
                                  0.5 * limit, 2000)
    size = float(np.linalg.norm(final))
    assert size < 1e-6, f'did not converge: ||x|| = {size:.2e}'
    return f'||x|| = {size:.1e} at h = 0.5 x limit'


def _g_gd_diverges(impl) -> str:
    curvature = np.array([4.0, 1.0])
    limit = 2.0 / (2.0 * curvature.max())
    grad = lambda x: 2.0 * curvature * x
    final = impl.gradient_descent(grad, np.array([5.0, -3.0]),
                                  1.5 * limit, 200)
    size = float(np.linalg.norm(final))
    assert (not np.isfinite(size)) or size > 1e3, (
        'gradient descent must diverge above 2/L; yours did not')
    return 'diverges above 2/L, as required'


def _g_jacobian(impl) -> str:
    f = lambda x: np.array([x[0] ** 2 + x[1], np.sin(x[2]) * x[0]])
    analytic = lambda x: np.array([[2 * x[0], 1.0, 0.0],
                                   [np.sin(x[2]), 0.0, np.cos(x[2]) * x[0]]])
    x = np.array([0.7, -1.3, 0.4])
    err = float(np.abs(impl.numeric_jacobian(f, x) - analytic(x)).max())
    assert err < 1e-6, f'max gap {err:.2e}'
    return f'matches analytic to {err:.1e}'


def _g_backprop(impl) -> str:
    rng = _rng(15)
    w1, b1 = rng.normal(size=(4, 3)), rng.normal(size=4)
    w2, b2 = rng.normal(size=(1, 4)), rng.normal(size=1)
    x, y = rng.normal(size=3), 0.4
    _, grads = impl.two_layer_backward(w1, b1, w2, b2, x, y)

    def loss_of(w1_, b1_, w2_, b2_):
        a1 = np.tanh(w1_ @ x + b1_)
        return 0.5 * float(((w2_ @ a1 + b2_ - y) ** 2).item())

    h = 1e-6
    numeric = np.zeros_like(w1)
    for i in range(w1.shape[0]):
        for j in range(w1.shape[1]):
            up, down = w1.copy(), w1.copy()
            up[i, j] += h
            down[i, j] -= h
            numeric[i, j] = (loss_of(up, b1, w2, b2) -
                             loss_of(down, b1, w2, b2)) / (2 * h)
    err = float(np.abs(grads['w1'] - numeric).max())
    assert err < 1e-6, f'dL/dW1 differs by {err:.2e}'
    return f'hand-derived gradients match numeric to {err:.1e}'


def _g_dimensionality(impl) -> str:
    for d, eps in ((2, 0.05), (10, 0.05), (100, 0.05)):
        got = impl.shell_fraction(d, eps)
        want = 1.0 - (1.0 - eps) ** d
        assert abs(got - want) < 1e-12, f'd={d}'
    high = impl.shell_fraction(100, 0.05)
    assert high > 0.99, 'at d=100 almost all volume is in the shell'
    return f'at d=100, {high:.3%} of the volume is within 5% of the surface'


def _g_entropy(impl) -> str:
    n = 8
    uniform = np.full(n, 1.0 / n)
    assert abs(impl.entropy(uniform) - np.log(n)) < 1e-12
    spike = np.zeros(n)
    spike[0] = 1.0
    assert abs(impl.entropy(spike)) < 1e-12
    return f'uniform = log {n}, point mass = 0'


def _g_kl(impl) -> str:
    rng = _rng(16)
    p = rng.random(6)
    p /= p.sum()
    assert abs(impl.kl_divergence(p, p)) < 1e-12, 'KL(p,p) must be 0'
    asymmetric = False
    for _ in range(1000):
        a, b = rng.random(6), rng.random(6)
        a, b = a / a.sum(), b / b.sum()
        kl_ab = impl.kl_divergence(a, b)
        assert kl_ab >= -1e-12, 'KL must be non-negative'
        if abs(kl_ab - impl.kl_divergence(b, a)) > 1e-6:
            asymmetric = True
    assert asymmetric, 'KL should not be symmetric'
    return 'non-negative over 1000 pairs, zero at p=q, asymmetric'


def _g_mle(impl) -> str:
    n, mu, sigma = 100_000, 1.75, 2.5
    samples = _rng(17).normal(loc=mu, scale=sigma, size=n)
    mu_hat, var_hat = impl.mle_gaussian(samples)
    standard_error = sigma / np.sqrt(n)
    assert abs(mu_hat - mu) < 3.0 * standard_error, 'mu outside 3 SE'
    assert abs(var_hat - sigma ** 2) < 0.05 * sigma ** 2, 'variance off'
    return f'mu_hat = {mu_hat:.4f} (true {mu}), var_hat = {var_hat:.4f}'


def _g_welford(impl) -> str:
    big = np.float64(1e9) + _rng(18).normal(size=50_000)
    _, variance = impl.welford(big)
    want = float(big.var(ddof=1))
    assert abs(variance - want) < 1e-6 * want, 'Welford disagrees with numpy'
    naive = float((big ** 2).mean() - big.mean() ** 2)
    return f'welford {variance:.4f} vs numpy {want:.4f}; naive gives {naive:.4f}'


def _g_bayes(impl) -> str:
    posterior = impl.bayes_posterior(0.99, 0.99, 0.001)
    assert abs(posterior - 0.0902) < 0.002, f'got {posterior:.4f}'
    return f'P(sick | positive) = {posterior:.3f} at 0.1% prevalence'


# ---- Part 2 gates ---------------------------------------------------- #

def _fit_predict(x_train, y_train, x_test):
    """Least-squares probe used by the leakage gates."""
    design = np.hstack([x_train, np.ones((x_train.shape[0], 1))])
    weights, *_ = np.linalg.lstsq(design, y_train, rcond=None)
    return np.hstack([x_test, np.ones((x_test.shape[0], 1))]) @ weights


def _r2(y_true, y_pred):
    """Coefficient of determination."""
    total = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / total


def _g_leakage(impl) -> str:
    rng = _rng(20260101)
    n, p, keep = 200, 2000, 20
    x = rng.normal(size=(n, p))
    y = rng.normal(size=n)                     # no signal whatsoever
    idx = rng.permutation(n)
    train, test = idx[:140], idx[140:]

    leaked_idx = np.argsort(np.abs(x.T @ (y - y.mean())))[-keep:]
    leaked = _r2(y[test], _fit_predict(x[train][:, leaked_idx], y[train],
                                       x[test][:, leaked_idx]))
    honest_idx = impl.select_features(x[train], y[train], keep)
    honest = _r2(y[test], _fit_predict(x[train][:, honest_idx], y[train],
                                       x[test][:, honest_idx]))
    assert honest < 0.2, f'honest R2 {honest:.3f} too high on pure noise'
    assert leaked > honest + 0.1, 'the leak should inflate the score'
    return f'leaked R2 {leaked:+.3f} vs honest R2 {honest:+.3f} on pure noise'


def _g_lookforward(impl) -> str:
    good = [(np.arange(0, 50), np.arange(50, 60))]
    impl.assert_no_lookforward(good)
    bad = [(np.arange(0, 60), np.arange(50, 70))]
    try:
        impl.assert_no_lookforward(bad)
    except AssertionError:
        return 'accepts a clean fold, rejects an overlapping one'
    raise AssertionError('a lookforward leak was not detected')


def _g_standardize(impl) -> str:
    rng = _rng(21)
    x_train = rng.normal(loc=5.0, scale=2.0, size=(200, 3))
    x_test = rng.normal(loc=50.0, scale=9.0, size=(80, 3))
    mean, std = impl.standardize_fit(x_train)
    assert np.allclose(mean, x_train.mean(axis=0), atol=1e-12)
    pooled = np.vstack([x_train, x_test]).mean(axis=0)
    assert not np.allclose(mean, pooled, atol=1e-3), (
        'the scaler appears to have seen the test rows')
    scaled = (x_train - mean) / std
    assert abs(float(scaled.mean())) < 1e-10
    return 'fitted on train only; train mean 0, sd 1 after scaling'


def _g_ridge(impl) -> str:
    rng = _rng(22)
    x = rng.normal(size=(300, 6))
    true_w = rng.normal(size=6)
    y = x @ true_w + 1.5 + 0.1 * rng.normal(size=300)
    zero = impl.ridge_fit(x, y, 0.0)
    design = np.hstack([x, np.ones((300, 1))])
    want, *_ = np.linalg.lstsq(design, y, rcond=None)
    assert np.allclose(zero, want, atol=1e-6), 'alpha=0 must equal OLS'
    strong = impl.ridge_fit(x, y, 1000.0)
    assert np.linalg.norm(strong[:-1]) < np.linalg.norm(zero[:-1]), (
        'a large alpha must shrink the coefficients')
    return 'alpha=0 matches OLS to 1e-6; alpha=1000 shrinks coefficients'


def _g_logistic(impl) -> str:
    rng = _rng(23)
    x = rng.normal(size=(200, 4))
    y = (x @ rng.normal(size=4) > 0).astype(np.float64)
    w = rng.normal(size=4)

    def loss(weights):
        z = x @ weights
        p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    analytic = impl.logistic_gradient(w, x, y)
    numeric = np.zeros_like(w)
    h = 1e-6
    for i in range(len(w)):
        up, down = w.copy(), w.copy()
        up[i] += h
        down[i] -= h
        numeric[i] = (loss(up) - loss(down)) / (2 * h)
    err = float(np.abs(analytic - numeric).max())
    assert err < 1e-6, f'gradient differs by {err:.2e}'
    return f'analytic gradient matches numeric to {err:.1e}'


def _g_gini(impl) -> str:
    assert abs(impl.gini_impurity(np.array([1, 1, 1, 1]))) < 1e-12
    assert abs(impl.gini_impurity(np.array([0, 1])) - 0.5) < 1e-12
    mixed = impl.gini_impurity(np.array([0, 0, 0, 1]))
    assert abs(mixed - 0.375) < 1e-12, f'got {mixed}'
    return 'pure 0.0, balanced 0.5, 3:1 split 0.375'


def _g_prf1(impl) -> str:
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 1, 0, 0, 0, 0])
    precision, recall, f1 = impl.precision_recall_f1(y_true, y_pred)
    assert abs(precision - 2 / 3) < 1e-12, f'precision {precision}'
    assert abs(recall - 2 / 3) < 1e-12, f'recall {recall}'
    assert abs(f1 - 2 / 3) < 1e-12, f'f1 {f1}'
    return 'precision, recall and F1 all 0.667 on the worked example'


def _g_bootstrap(impl) -> str:
    rng = _rng(24)
    y = rng.normal(size=400)
    pred = y + rng.normal(scale=0.5, size=400)
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    low, high = impl.bootstrap_ci(rmse, y, pred, 400, 0.95, 1)
    point = rmse(y, pred)
    assert low <= point <= high, 'interval must bracket the point estimate'
    again = impl.bootstrap_ci(rmse, y, pred, 400, 0.95, 1)
    assert (low, high) == again, 'same seed must give the same interval'
    return f'95% CI [{low:.4f}, {high:.4f}] around {point:.4f}, reproducible'


def _g_kmeans(impl) -> str:
    rng = _rng(25)
    blobs = np.vstack([rng.normal(loc=c, scale=0.35, size=(60, 2))
                       for c in ([0, 0], [4, 4], [0, 5])])
    _, _, history = impl.kmeans(blobs, 3, 30, seed=0)
    diffs = np.diff(history)
    assert np.all(diffs <= 1e-8), 'the k-means objective must never increase'
    return f'objective fell {history[0]:.1f} -> {history[-1]:.1f}, monotone'


# ---- Part 3 gates ---------------------------------------------------- #

def _g_softmax(impl) -> str:
    z = np.array([1000.0, 1001.0, 1002.0])
    p = impl.stable_softmax(z)
    assert np.all(np.isfinite(p)), 'softmax overflowed at z ~ 1000'
    assert abs(float(p.sum()) - 1.0) < 1e-12, 'probabilities must sum to 1'
    shifted = impl.stable_softmax(z - 1000.0)
    assert np.allclose(p, shifted, atol=1e-12), 'must be shift-invariant'
    return 'finite at z=1000, sums to 1, shift-invariant'


def _g_ce_grad(impl) -> str:
    rng = _rng(26)
    logits = rng.normal(size=5)
    one_hot = np.zeros(5)
    one_hot[2] = 1.0

    def loss(z):
        shifted = z - z.max()
        return float(-np.log(np.exp(shifted)[2] / np.exp(shifted).sum()))

    analytic = impl.cross_entropy_grad(logits, one_hot)
    numeric = np.zeros(5)
    h = 1e-6
    for i in range(5):
        up, down = logits.copy(), logits.copy()
        up[i] += h
        down[i] -= h
        numeric[i] = (loss(up) - loss(down)) / (2 * h)
    err = float(np.abs(analytic - numeric).max())
    assert err < 1e-6, f'gradient differs by {err:.2e}'
    return f'p - y matches the numeric gradient to {err:.1e}'


def _g_conv(impl) -> str:
    rng = _rng(27)
    x = rng.normal(size=(12, 12))
    kernel = rng.normal(size=(3, 3))
    got = impl.conv2d(x, kernel)
    want = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            want[i, j] = float((x[i:i + 3, j:j + 3] * kernel).sum())
    err = float(np.abs(got - want).max())
    assert err < 1e-10, f'differs from the direct loop by {err:.2e}'
    return f'im2col matches the direct convolution to {err:.1e}'


def _g_attention(impl) -> str:
    rng = _rng(28)
    q, k, v = (rng.normal(size=(4, 8)), rng.normal(size=(6, 8)),
               rng.normal(size=(6, 3)))
    got = impl.attention(q, k, v)
    scores = (q @ k.T) / np.sqrt(8)
    shifted = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
    want = weights @ v
    assert got.shape == (4, 3), f'shape {got.shape}, expected (4, 3)'
    err = float(np.abs(got - want).max())
    assert err < 1e-10, f'differs by {err:.2e}'
    assert np.allclose(weights.sum(axis=-1), 1.0), 'rows must sum to 1'
    return f'matches the reference to {err:.1e}; attention rows sum to 1'


def _g_training_step(impl) -> str:
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 1)
    optimiser = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = torch.nn.MSELoss()
    xb = torch.randn(32, 3)
    yb = (xb @ torch.tensor([[1.0], [-2.0], [0.5]])) + 0.3
    first = impl.torch_training_step(model, xb, yb, optimiser, loss_fn)
    for _ in range(100):
        last = impl.torch_training_step(model, xb, yb, optimiser, loss_fn)
    assert last < first, f'loss did not fall: {first:.4f} -> {last:.4f}'
    assert last < 0.05, f'loss only reached {last:.4f}'
    return f'loss {first:.4f} -> {last:.5f} in 100 steps'


# ---- Part 4 gates ---------------------------------------------------- #

def _g_acf(impl) -> str:
    rng = _rng(29)
    n, phi = 4000, 0.7
    series = np.zeros(n)
    for t in range(1, n):
        series[t] = phi * series[t - 1] + rng.normal()
    got = impl.acf(series, 5)
    assert abs(got[0] - 1.0) < 1e-12, 'lag 0 must be exactly 1'
    for lag in (1, 2, 3):
        assert abs(got[lag] - phi ** lag) < 0.07, f'lag {lag}'
    return f'AR(0.7): acf(1) = {got[1]:.3f}, acf(2) = {got[2]:.3f}'


def _g_decompose(impl) -> str:
    rng = _rng(30)
    period, n = 12, 240
    t = np.arange(n)
    seasonal = 5.0 * np.sin(2 * np.pi * t / period)
    series = 10.0 + 0.05 * t + seasonal + rng.normal(scale=0.3, size=n)
    trend, season, residual = impl.additive_decompose(series, period)
    assert abs(float(residual.mean())) < 0.5, 'residual mean should be ~0'
    recon = float(np.abs(trend + season + residual - series).max())
    assert recon < 1e-8, f'components do not re-sum: {recon:.2e}'
    return f'residual mean {residual.mean():+.3f}; components re-sum exactly'


def _g_rolling(impl) -> str:
    folds = impl.rolling_window_cv(120, 4, 10, 50)
    assert len(folds) == 4, f'expected 4 folds, got {len(folds)}'
    sizes = {len(train) for train, _ in folds}
    assert sizes == {50}, f'training window must stay 50, got {sizes}'
    Solutions.assert_no_lookforward(folds)
    return '4 folds, constant 50-point history, no lookforward'


def _g_naive_scale(impl) -> str:
    rng = _rng(31)
    series = np.cumsum(rng.normal(size=400))
    scale = impl.naive_scale(series[:300], 1)
    assert scale > 0
    try:
        impl.naive_scale(np.ones(50), 1)
    except ValueError:
        return f'scale {scale:.4f}; constant series correctly rejected'
    raise AssertionError('a constant training series must raise')


def _g_mase(impl) -> str:
    rng = _rng(32)
    series = np.cumsum(rng.normal(size=400))
    y_train, y_true = series[:300], series[300:]
    naive_pred = np.concatenate([[series[299]], y_true[:-1]])
    score = impl.mase(y_true, naive_pred, y_train, 1)
    assert abs(score - 1.0) < 0.35, f'naive forecast should score ~1, got {score:.3f}'
    offset = 273.15
    shifted = impl.mase(y_true + offset, naive_pred + offset,
                        y_train + offset, 1)
    assert abs(score - shifted) < 1e-10, 'MASE must be shift-invariant'
    return f'naive scores {score:.3f}; unchanged under a unit shift'


def _g_ape_zero(impl) -> str:
    fine = impl.absolute_percentage_error(np.array([1.0, 2.0]),
                                          np.array([1.1, 1.8]))
    assert np.all(np.isfinite(fine))
    try:
        impl.absolute_percentage_error(np.array([1.0, 0.0, 2.0]),
                                       np.array([1.0, 0.5, 2.0]))
    except (ValueError, ZeroDivisionError):
        return 'refuses zero targets instead of returning infinity'
    raise AssertionError('APE at a zero target must raise [TSP p.144]')


def _g_conformal(impl) -> str:
    rng = _rng(33)
    n_cal, n_test = 1000, 4000
    cal_residuals = np.abs(rng.normal(scale=2.0, size=n_cal))
    details = []
    for alpha in (0.2, 0.1, 0.05):
        q = impl.split_conformal(cal_residuals, alpha)
        test_residuals = np.abs(rng.normal(scale=2.0, size=n_test))
        covered = float(np.mean(test_residuals <= q))
        assert abs(covered - (1 - alpha)) < 0.03, (
            f'alpha={alpha}: coverage {covered:.3f}, target {1 - alpha:.2f}')
        details.append(f'{1 - alpha:.0%}:{covered:.3f}')
    wide = impl.split_conformal(cal_residuals, 0.2)
    narrow = impl.split_conformal(cal_residuals, 0.05)
    assert narrow > wide, 'higher confidence must give a wider interval'
    return 'coverage ' + ', '.join(details) + '; width grows with confidence'


def _g_eig_2x2(impl) -> str:
    a = np.array([[4.0, 1.0], [2.0, 3.0]])
    high, low = impl.eigenvalues_2x2(a)
    assert abs(high - 5.0) < 1e-12 and abs(low - 2.0) < 1e-12, (high, low)
    assert abs((high + low) - np.trace(a)) < 1e-12, 'trace must equal the sum'
    assert abs(high * low - np.linalg.det(a)) < 1e-12, 'det must equal product'
    return 'eigenvalues 5 and 2; trace and determinant identities hold'


def _g_singular_values(impl) -> str:
    a = np.array([[3.0, 0.0], [4.0, 5.0]])
    got = np.sort(np.asarray(impl.singular_values(a)))[::-1]
    want = np.sort(np.linalg.svd(a, compute_uv=False))[::-1]
    assert np.allclose(got, want, atol=1e-10), f'{got} vs {want}'
    assert abs(float(np.prod(got)) - abs(float(np.linalg.det(a)))) < 1e-10, (
        'the product of singular values must equal |det|')
    return f'sigma = ({got[0]:.4f}, {got[1]:.4f}); product equals |det| = 15'


def _g_threshold(impl) -> str:
    got = impl.optimal_threshold(5.0, 500.0)
    assert abs(got - 5.0 / 505.0) < 1e-12, f'got {got}'
    assert abs(impl.optimal_threshold(1.0, 1.0) - 0.5) < 1e-12, (
        'equal costs must give the usual 0.5')
    return f'threshold {got:.4f} when a miss costs 100x a false alarm'


def _g_soft_threshold(impl) -> str:
    assert impl.soft_threshold(0.3, 0.5) == 0.0, 'must be exactly zero'
    assert impl.soft_threshold(-0.3, 0.5) == 0.0, 'must be exactly zero'
    assert abs(impl.soft_threshold(2.0, 0.5) - 1.5) < 1e-12
    assert abs(impl.soft_threshold(-2.0, 0.5) + 1.5) < 1e-12
    ridge = 2.0 / (1.0 + 2 * 0.5)
    assert ridge != 0.0, 'the L2 comparison should never be exactly zero'
    return 'exact zeros below alpha; shrinks by alpha above (L2 never zeroes)'


def _g_gini_gain(impl) -> str:
    parent = np.array([1] * 50 + [0] * 50)
    left_a = np.array([1] * 40 + [0] * 10)
    right_a = np.array([1] * 10 + [0] * 40)
    gain_a = impl.gini_gain(parent, left_a, right_a)
    assert abs(gain_a - 0.18) < 1e-9, f'split A gain {gain_a:.4f}, expected 0.18'
    left_b = np.array([1] * 10)
    right_b = np.array([1] * 40 + [0] * 50)
    gain_b = impl.gini_gain(parent, left_b, right_b)
    assert abs(gain_b - 0.05555555555) < 1e-6, f'split B gain {gain_b:.4f}'
    assert gain_a > gain_b, 'the balanced split should win'
    return f'split A {gain_a:.4f} beats split B {gain_b:.4f}, as worked'


def _g_conv_size(impl) -> str:
    assert impl.conv_output_size(32, 3, 1, 1, 1) == 32, 'same-padding case'
    assert impl.conv_output_size(32, 3, 2, 1, 1) == 16, 'stride-2 case'
    assert impl.conv_output_size(28, 5, 1, 0, 1) == 24, 'valid case'
    assert impl.conv_output_size(32, 3, 1, 2, 2) == 32, 'dilated case'
    return '32->32 (same), 32->16 (stride 2), 28->24 (valid), dilation handled'


def _g_ar1_acf(impl) -> str:
    phi = 0.7
    theory = impl.ar1_acf(phi, 5)
    assert abs(theory[0] - 1.0) < 1e-12, 'lag 0 must be 1'
    for k in range(1, 6):
        assert abs(theory[k] - phi ** k) < 1e-12, f'lag {k}'
    rng = _rng(40)
    n = 20_000
    series = np.zeros(n)
    for t in range(1, n):
        series[t] = phi * series[t - 1] + rng.normal()
    empirical = Solutions.acf(series, 3)
    assert np.all(np.abs(empirical[1:4] - theory[1:4]) < 0.05), (
        'the simulated ACF should match the theory')
    return f'theory {theory[1]:.3f}, {theory[2]:.3f} matches simulation'


def _g_winkler(impl) -> str:
    inside = impl.winkler_score(5.0, 4.0, 6.0, 0.1)
    assert abs(inside - 2.0) < 1e-12, 'inside, the score is the width'
    below = impl.winkler_score(3.0, 4.0, 6.0, 0.1)
    assert abs(below - (2.0 + 20.0 * 1.0)) < 1e-12, f'got {below}'
    above = impl.winkler_score(7.0, 4.0, 6.0, 0.1)
    assert abs(above - (2.0 + 20.0 * 1.0)) < 1e-12, f'got {above}'
    wide = impl.winkler_score(5.0, 0.0, 100.0, 0.1)
    assert wide > inside, 'a needlessly wide interval must score worse'
    return 'width when covered; width + penalty when missed; punishes padding'


EXERCISES: List[Exercise] = [
    Exercise('1.1', 1, 'l_p norms match NumPy', 'Ch 2 · MML p.49', _g_norm),
    Exercise('1.2', 1, 'Vectorisation speedup', 'Ch 1 · MML p.37',
             _g_vectorisation),
    Exercise('1.3', 1, 'Rank detects collinearity', 'Ch 1 · MML p.15',
             _g_rank),
    Exercise('2.1', 1, 'Gram-Schmidt orthonormality', 'Ch 2 · MML p.89',
             _g_gram_schmidt),
    Exercise('2.2', 1, 'Classical vs modified on a Hilbert matrix',
             'Ch 2 · MML p.67', _g_gram_schmidt_stability),
    Exercise('2.3', 1, 'Projection is idempotent and symmetric', 'Ch 2',
             _g_projection),
    Exercise('3.1', 1, 'Rank-nullity theorem', 'Ch 3 · MML p.127',
             _g_rank_nullity),
    Exercise('3.2', 1, 'Rotations preserve length', 'Ch 3 · MML p.134',
             _g_rotation),
    Exercise('4.1', 1, 'LU decomposition: PA = LU', 'Ch 4 · MML p.171',
             _g_lu),
    Exercise('4.2', 1, 'Solving from the factorisation', 'Ch 4', _g_lu_solve),
    Exercise('4.3', 1, 'Determinant via LU', 'Ch 4 · MML p.182', _g_det),
    Exercise('4.4', 1, 'Solve beats explicit inversion', 'Ch 4 · MML p.177',
             _g_conditioning),
    Exercise('5.1', 1, 'Power iteration', 'Ch 5 · MML p.217',
             _g_power_iteration),
    Exercise('5.2', 1, 'Eckart-Young low-rank error', 'Ch 5 · MML p.207',
             _g_eckart_young),
    Exercise('5.3', 1, 'PCA from the SVD', 'Ch 5', _g_pca),
    Exercise('5.4', 1, 'Eigenvalues of a 2x2 by hand', 'Ch 5 · MML p.192',
             _g_eig_2x2),
    Exercise('5.5', 1, 'Singular values from AtA', 'Ch 5 · MML p.207',
             _g_singular_values),
    Exercise('6.1', 1, 'The U-shaped differentiation error', 'Ch 6',
             _g_central_difference),
    Exercise('6.2', 1, 'Gradient descent converges below 2/L',
             'Ch 6 · MML p.381', _g_gd_converges),
    Exercise('6.3', 1, 'Gradient descent diverges above 2/L',
             'Ch 6 · MML p.389', _g_gd_diverges),
    Exercise('7.1', 1, 'Numeric Jacobian', 'Ch 7 · MML p.460', _g_jacobian),
    Exercise('7.2', 1, 'Backpropagation by hand', 'Ch 7', _g_backprop),
    Exercise('7.3', 1, 'Curse of dimensionality', 'Ch 7 · MML p.443',
             _g_dimensionality),
    Exercise('8.1', 1, 'Entropy', 'Ch 8 · MML p.606', _g_entropy),
    Exercise('8.2', 1, 'KL divergence properties', 'Ch 8 · MML p.606', _g_kl),
    Exercise('8.3', 1, 'Maximum likelihood recovers parameters',
             'Ch 8 · MML p.619', _g_mle),
    Exercise('8.4', 1, 'Welford beats the naive variance', 'Ch 8 · MML p.589',
             _g_welford),
    Exercise('8.5', 1, 'Bayes and the base rate', 'Ch 8 · MML p.515',
             _g_bayes),

    Exercise('9.1', 2, 'The threshold follows from the costs', 'Ch 9',
             _g_threshold),
    Exercise('10.1', 2, 'Leakage: selection inside the fold',
             'Ch 10 · TSP p.158', _g_leakage),
    Exercise('10.2', 2, 'Lookforward leaks are detected',
             'Ch 10 · TSP p.161', _g_lookforward),
    Exercise('11.1', 2, 'Scalers are fitted on train only', 'Ch 11',
             _g_standardize),
    Exercise('12.1', 2, 'Ridge regression', 'Ch 12', _g_ridge),
    Exercise('12.2', 2, 'Logistic gradient is p - y', 'Ch 12', _g_logistic),
    Exercise('12.3', 2, 'Soft-threshold: why L1 selects', 'Ch 12',
             _g_soft_threshold),
    Exercise('13.1', 2, 'Gini impurity', 'Ch 13', _g_gini),
    Exercise('13.2', 2, 'Choosing between two splits', 'Ch 13',
             _g_gini_gain),
    Exercise('14.1', 2, 'Precision, recall, F1', 'Ch 14', _g_prf1),
    Exercise('14.2', 2, 'Bootstrap confidence intervals', 'Ch 14',
             _g_bootstrap),
    Exercise('15.1', 2, 'k-means objective never increases', 'Ch 15',
             _g_kmeans),

    Exercise('17.1', 3, 'Numerically stable softmax', 'Ch 17 · PTG p.37',
             _g_softmax),
    Exercise('17.2', 3, 'Cross-entropy gradient', 'Ch 17 · PTG p.40',
             _g_ce_grad),
    Exercise('19.1', 3, 'Convolution via im2col', 'Ch 19 · PTG p.128',
             _g_conv),
    Exercise('19.2', 3, 'Convolution output arithmetic',
             'Ch 19 · PTG p.129', _g_conv_size),
    Exercise('21.1', 3, 'Scaled dot-product attention', 'Ch 21 · PTG p.320',
             _g_attention),
    Exercise('18.1', 3, 'A training step that actually trains',
             'Ch 18 · PTG p.62', _g_training_step, needs_torch=True),

    Exercise('22.1', 4, 'Autocorrelation of an AR(1)', 'Ch 22 · TSP p.84',
             _g_acf),
    Exercise('22.2', 4, 'Additive decomposition', 'Ch 22 · TSP p.66',
             _g_decompose),
    Exercise('22.3', 4, 'The ACF of an AR(1) is phi^k', 'Ch 22 · TSP p.84',
             _g_ar1_acf),
    Exercise('23.1', 4, 'Rolling-origin folds', 'Ch 23 · TSP p.134',
             _g_rolling),
    Exercise('23.2', 4, 'The naive scale, on train only',
             'Ch 23 · TSP p.150', _g_naive_scale),
    Exercise('23.3', 4, 'MASE is scale invariant', 'Ch 23 · TSP p.143',
             _g_mase),
    Exercise('23.4', 4, 'APE refuses zero targets', 'Ch 23 · TSP p.144',
             _g_ape_zero),
    Exercise('25.1', 4, 'Conformal coverage', 'Ch 25 · TSP p.325',
             _g_conformal),
    Exercise('25.2', 4, 'Winkler interval score', 'Ch 25 · TSP p.346',
             _g_winkler),
]

PART_NAMES = {
    1: 'Part I  - The Mathematical Engine',
    2: 'Part II - Learning from Data',
    3: 'Part III- Neural Networks in PyTorch',
    4: 'Part IV - Sequences and Uncertainty',
}


def run_one(exercise: Exercise, impl: type) -> Tuple[str, str]:
    """Runs one exercise and classifies the outcome.

    Args:
        exercise: The exercise to run.
        impl: The implementation namespace, Student or Solutions.

    Returns:
        A tuple (status, detail).
    """
    if exercise.needs_torch and not HAS_TORCH:
        return SKIP, 'PyTorch is not installed; this gate cannot run'
    try:
        detail = exercise.check(impl)
        return PASS, detail or ''
    except NotImplementedYet as todo:
        return TODO, f'not implemented yet ({todo})'
    except AssertionError as failure:
        return FAIL, str(failure) or 'gate failed'
    except Exception as error:                  # noqa: BLE001 - reported
        return FAIL, f'{type(error).__name__}: {error}'


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point.

    Args:
        argv: Optional argument vector, for testing.

    Returns:
        0 when nothing failed and nothing was left TODO.
    """
    parser = argparse.ArgumentParser(
        description='Exercises for Machine Learning: The Working Text')
    parser.add_argument('--part', type=int, choices=sorted(PART_NAMES),
                        help='run only one part')
    parser.add_argument('--exercise', help='run one exercise, e.g. 5.2')
    parser.add_argument('--solutions', action='store_true',
                        help='run the reference implementations')
    parser.add_argument('--list', action='store_true',
                        help='list the exercises and exit')
    parser.add_argument('--verbose', action='store_true',
                        help='show the gate detail for passing exercises')
    parser.add_argument('--colab', action='store_true',
                        help='how to work these exercises in Google Colab')
    args = parser.parse_args(argv)

    selected = EXERCISES
    if args.part:
        selected = [e for e in selected if e.part == args.part]
    if args.exercise:
        selected = [e for e in selected if e.key == args.exercise]
        if not selected:
            print(f'no exercise {args.exercise!r}; try --list')
            return 2

    if args.colab:
        print('\nWorking these exercises in Google Colab')
        print('=' * 46)
        print('The notebooks are self-contained: no clone, no data upload,')
        print('no pip install. Colab already has NumPy and PyTorch.')
        print('\n  1. Open https://colab.research.google.com')
        print('  2. File -> Upload notebook, and choose one of:')
        for part in sorted(NOTEBOOKS):
            count = sum(1 for e in EXERCISES if e.part == part)
            print(f'       {NOTEBOOKS[part]:<46} {count:>2} exercises')
        print('  3. Run the Setup cell, then work down the notebook.')
        print('\nIf you push this repository to GitHub you can open them')
        print('directly instead, with no upload step:')
        print('  https://colab.research.google.com/github/<user>/<repo>'
              '/blob/main/ml-mastery/notebooks/<file>.ipynb')
        print('\nRegenerate them after editing this file:')
        print('  python notebooks/make_notebooks.py')
        print('  python notebooks/validate_notebooks.py')
        return 0

    if args.list:
        for part in sorted(PART_NAMES):
            rows = [e for e in selected if e.part == part]
            if not rows:
                continue
            print(f'\n{PART_NAMES[part]}')
            print(f'  Colab notebook: {NOTEBOOKS[part]}')
            for exercise in rows:
                flag = ' [needs torch]' if exercise.needs_torch else ''
                print(f'  {exercise.key:<5} {exercise.title:<44} '
                      f'{exercise.anchor}{flag}')
        print(f'\n{len(selected)} exercises')
        return 0

    impl = Solutions if args.solutions else Student
    source = 'reference solutions' if args.solutions else 'your implementations'
    print(f'\nMachine Learning: The Working Text -- exercises')
    print(f'running {len(selected)} exercise(s) against {source}')
    if not HAS_TORCH:
        print('note: PyTorch absent; torch exercises will report SKIP')
    print('=' * 78)

    counts = {PASS: 0, FAIL: 0, TODO: 0, SKIP: 0}
    current_part = None
    for exercise in selected:
        if exercise.part != current_part:
            current_part = exercise.part
            print(f'\n{PART_NAMES[current_part]}')
            print('-' * 78)
        status, detail = run_one(exercise, impl)
        counts[status] += 1
        marker = {PASS: '[PASS]', FAIL: '[FAIL]', TODO: '[TODO]',
                  SKIP: '[SKIP]'}[status]
        print(f'  {marker} {exercise.key:<5} {exercise.title}')
        if detail and (status != PASS or args.verbose):
            print(f'         {detail}')
        elif detail and status == PASS and not args.verbose:
            pass

    print('\n' + '=' * 78)
    print(f'passed {counts[PASS]}   failed {counts[FAIL]}   '
          f'todo {counts[TODO]}   skipped {counts[SKIP]}')
    if counts[TODO]:
        print(f'\n{counts[TODO]} exercise(s) are still TODO. A TODO is not a '
              f'pass:\nimplement the stubs in the STUDENT IMPLEMENTATIONS '
              f'section and run again.')
    if counts[SKIP]:
        print(f'\n{counts[SKIP]} exercise(s) were SKIPPED for a missing '
              f'dependency.\nThose results are UNVERIFIED, which is not the '
              f'same as passing.')
    if counts[FAIL]:
        print(f'\n{counts[FAIL]} gate(s) FAILED. Read the detail above: the '
              f'threshold came\nfrom the book, so the code is what needs to '
              f'change.')
    if not counts[FAIL] and not counts[TODO]:
        print('\nAll runnable gates passed.')
    return 0 if not counts[FAIL] and not counts[TODO] else 1


if __name__ == '__main__':
    sys.exit(main())
