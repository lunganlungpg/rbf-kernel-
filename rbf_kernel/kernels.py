"""Core RBF kernel functions and gamma heuristics.

The RBF (Gaussian) kernel is

    K(x, x') = exp(-gamma * ||x - x'||^2),    gamma = 1 / (2 * sigma^2)

where ``sigma`` is the kernel's length scale ("width").
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _as_2d(X: ArrayLike, name: str = "X") -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D, got shape {X.shape}")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return X


def _check_gamma(gamma: float) -> float:
    gamma = float(gamma)
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError(f"gamma must be a positive finite number, got {gamma}")
    return gamma


def pairwise_sq_dists(X: ArrayLike, Y: ArrayLike | None = None) -> np.ndarray:
    """Squared Euclidean distances between the rows of ``X`` and ``Y``.

    Uses the expansion ``||x - y||^2 = ||x||^2 + ||y||^2 - 2 x.y`` so the whole
    matrix is computed with one matrix multiply, then clips tiny negative values
    caused by floating-point round-off.

    Returns an array of shape ``(n_samples_X, n_samples_Y)``.
    """
    X = _as_2d(X, "X")
    Y = X if Y is None else _as_2d(Y, "Y")
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"X and Y must have the same number of features, got {X.shape[1]} and {Y.shape[1]}"
        )
    x_sq = np.einsum("ij,ij->i", X, X)[:, None]
    y_sq = np.einsum("ij,ij->i", Y, Y)[None, :]
    d2 = x_sq + y_sq - 2.0 * (X @ Y.T)
    np.maximum(d2, 0.0, out=d2)
    if Y is X:
        np.fill_diagonal(d2, 0.0)
    return d2


def rbf_kernel(X: ArrayLike, Y: ArrayLike | None = None, gamma: float = 1.0) -> np.ndarray:
    """Compute the RBF (Gaussian) kernel matrix ``K[i, j] = exp(-gamma ||X_i - Y_j||^2)``.

    Parameters
    ----------
    X : array of shape (n_samples_X, n_features)
    Y : array of shape (n_samples_Y, n_features), optional
        Defaults to ``X`` (gives the square Gram matrix).
    gamma : float
        Kernel coefficient. Larger values mean a narrower kernel.

    Examples
    --------
    >>> import numpy as np
    >>> K = rbf_kernel(np.array([[0.0], [1.0]]), gamma=1.0)
    >>> np.round(K, 4)
    array([[1.    , 0.3679],
           [0.3679, 1.    ]])
    """
    gamma = _check_gamma(gamma)
    return np.exp(-gamma * pairwise_sq_dists(X, Y))


def sigma_to_gamma(sigma: float) -> float:
    """Convert a Gaussian length scale ``sigma`` to ``gamma = 1 / (2 sigma^2)``."""
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"sigma must be a positive finite number, got {sigma}")
    return 1.0 / (2.0 * sigma**2)


def gamma_to_sigma(gamma: float) -> float:
    """Convert ``gamma`` back to the Gaussian length scale ``sigma = sqrt(1 / (2 gamma))``."""
    return float(np.sqrt(1.0 / (2.0 * _check_gamma(gamma))))


def gamma_scale(X: ArrayLike) -> float:
    """scikit-learn's ``gamma='scale'`` heuristic: ``1 / (n_features * X.var())``.

    This is the default for :class:`sklearn.svm.SVC` and is a sensible first guess
    when your features are standardized (it equals ``1 / n_features`` then).
    """
    X = _as_2d(X)
    var = X.var()
    if var == 0:
        raise ValueError("X has zero variance; gamma='scale' is undefined")
    return 1.0 / (X.shape[1] * var)


def median_heuristic_gamma(
    X: ArrayLike, max_samples: int = 1000, random_state: int | None = 0
) -> float:
    """The *median heuristic*: pick ``sigma`` = median pairwise distance.

    Returns ``gamma = 1 / (2 * median(||x_i - x_j||)^2)``. It adapts to the actual
    spread of your data and is widely used for kernel two-sample tests and
    Gaussian processes. For speed, at most ``max_samples`` rows are used.
    """
    X = _as_2d(X)
    if X.shape[0] < 2:
        raise ValueError("Need at least 2 samples for the median heuristic")
    if X.shape[0] > max_samples:
        rng = np.random.default_rng(random_state)
        X = X[rng.choice(X.shape[0], size=max_samples, replace=False)]
    d2 = pairwise_sq_dists(X)
    iu = np.triu_indices_from(d2, k=1)
    med = np.median(np.sqrt(d2[iu]))
    if med == 0:
        raise ValueError("Median pairwise distance is zero; data has too many duplicates")
    return sigma_to_gamma(med)
