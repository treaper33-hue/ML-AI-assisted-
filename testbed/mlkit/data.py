"""The shared input field.

Every experiment loads data through `load_dataset`. Datasets are generated
from a fixed seed and fingerprinted, so two methodologies are always compared
on byte-identical inputs.

Anchors:
    Design-matrix view of data: [MML p.110]
    Reading data as a schema: [SKL ch.3 "Foundations"]
    Time-series structures: [TSP p.103-p.106]
"""

import dataclasses
import hashlib
from typing import Any, Callable, Dict, Tuple

import numpy as np

DEFAULT_SEED = 20260101


@dataclasses.dataclass(frozen=True)
class DatasetCard:
    """Provenance record that travels with every dataset.

    Attributes:
        name: Registry name of the dataset.
        n_samples: Number of rows.
        n_features: Number of feature columns.
        seed: Seed used to generate the data.
        checksum: SHA-256 fingerprint of the arrays.
        structure: One of 'tabular', 'univariate', 'panel'.
        provenance: How the data came to exist.
        notes: Anything a reviewer needs to know.
    """

    name: str
    n_samples: int
    n_features: int
    seed: int
    checksum: str
    structure: str
    provenance: str
    notes: str = ''

    def as_dict(self) -> Dict[str, Any]:
        """Returns the card as a JSON-serialisable dictionary."""
        return dataclasses.asdict(self)


def fingerprint(*arrays: np.ndarray) -> str:
    """Computes a stable SHA-256 fingerprint over one or more arrays.

    Args:
        *arrays: Arrays to fingerprint, in order.

    Returns:
        The first 16 hex characters of the digest.
    """
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode('utf-8'))
        digest.update(str(contiguous.dtype).encode('utf-8'))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()[:16]


def _synthetic_linear(
    seed: int,
    n_samples: int = 2000,
    n_features: int = 8,
    noise: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generates a linear regression problem with a known optimum.

    The true coefficients are returned in the extras dictionary so that an
    auditor can check a fitted model against ground truth rather than
    against another model.

    Args:
        seed: Seed for the random generator.
        n_samples: Number of rows.
        n_features: Number of feature columns.
        noise: Standard deviation of the additive Gaussian noise.

    Returns:
        A tuple of (features, targets, extras).
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_samples, n_features))
    true_coef = rng.normal(size=n_features) * 2.0
    true_bias = 0.5
    y = x @ true_coef + true_bias + rng.normal(scale=noise, size=n_samples)
    extras = {'true_coef': true_coef, 'true_bias': true_bias, 'noise': noise}
    return x, y, extras


def _synthetic_binary(
    seed: int,
    n_samples: int = 2000,
    n_features: int = 8,
    class_balance: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generates an imbalanced binary classification problem.

    Args:
        seed: Seed for the random generator.
        n_samples: Number of rows.
        n_features: Number of feature columns.
        class_balance: Target share of the positive class.

    Returns:
        A tuple of (features, labels, extras).
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_samples, n_features))
    weights = rng.normal(size=n_features)
    logits = x @ weights
    threshold = np.quantile(logits, 1.0 - class_balance)
    y = (logits > threshold).astype(np.int64)
    extras = {'weights': weights, 'positive_rate': float(y.mean())}
    return x, y, extras


def _synthetic_seasonal(
    seed: int,
    n_samples: int = 730,
    period: int = 7,
    trend: float = 0.02,
    noise: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generates a univariate series with trend and seasonality.

    The additive form matches the decomposition described at [TSP p.66].

    Args:
        seed: Seed for the random generator.
        n_samples: Length of the series.
        period: Seasonal period in steps.
        trend: Slope of the linear trend per step.
        noise: Standard deviation of the additive noise.

    Returns:
        A tuple of (time index as a column, series values, extras).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples, dtype=np.float64)
    seasonal = 3.0 * np.sin(2.0 * np.pi * t / period)
    level = 10.0 + trend * t
    y = level + seasonal + rng.normal(scale=noise, size=n_samples)
    extras = {'period': period, 'trend': trend, 'noise': noise}
    return t.reshape(-1, 1), y, extras


_REGISTRY: Dict[str, Callable[..., Tuple[np.ndarray, np.ndarray, Dict]]] = {
    'synthetic_linear': _synthetic_linear,
    'synthetic_binary': _synthetic_binary,
    'synthetic_seasonal': _synthetic_seasonal,
}

_STRUCTURE: Dict[str, str] = {
    'synthetic_linear': 'tabular',
    'synthetic_binary': 'tabular',
    'synthetic_seasonal': 'univariate',
}


def available_datasets() -> Tuple[str, ...]:
    """Returns the names registered in the shared input field."""
    return tuple(sorted(_REGISTRY))


def load_dataset(
    name: str,
    seed: int = DEFAULT_SEED,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, DatasetCard, Dict[str, Any]]:
    """Loads a dataset through the shared input field.

    Args:
        name: Registry name; see `available_datasets`.
        seed: Seed recorded on the dataset card.
        **kwargs: Generator-specific parameters.

    Returns:
        A tuple of (features, target, card, extras).

    Raises:
        KeyError: If `name` is not registered.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f'unknown dataset {name!r}; available: {available_datasets()}'
        )
    x, y, extras = _REGISTRY[name](seed=seed, **kwargs)
    card = DatasetCard(
        name=name,
        n_samples=int(x.shape[0]),
        n_features=int(x.shape[1]),
        seed=seed,
        checksum=fingerprint(x, y),
        structure=_STRUCTURE[name],
        provenance=f'generated by mlkit.data.{_REGISTRY[name].__name__}',
        notes='deterministic given (name, seed, kwargs)',
    )
    return x, y, card, extras
