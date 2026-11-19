"""Scalable approximations of the RBF kernel.

An exact kernel matrix costs O(n^2) memory and kernel solvers cost up to O(n^3)
time. Both classes here build an explicit, finite feature map ``z(x)`` with
``z(x) . z(x') ~= K(x, x')`` so you can use fast *linear* models instead.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .kernels import _as_2d, _check_gamma, rbf_kernel


class RandomFourierFeatures(TransformerMixin, BaseEstimator):
    """Random Fourier Features (Rahimi & Recht, 2007).

    By Bochner's theorem the RBF kernel is the Fourier transform of a Gaussian,
    so sampling ``w ~ N(0, 2*gamma*I)`` and ``b ~ U(0, 2*pi)`` gives

        z(x) = sqrt(2 / D) * cos(W x + b),   E[z(x) . z(x')] = K(x, x').

    The approximation error shrinks like ``1 / sqrt(n_components)``.

    Parameters
    ----------
    gamma : float, default=1.0
    n_components : int, default=500
        Number of random features ``D``.
    random_state : int or None, default=None
    """

    def __init__(self, gamma: float = 1.0, n_components: int = 500, random_state=None):
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X: ArrayLike, y=None):
        X = _as_2d(X)
        gamma = _check_gamma(self.gamma)
        if self.n_components < 1:
            raise ValueError("n_components must be >= 1")
        rng = np.random.default_rng(self.random_state)
        self.weights_ = rng.normal(0.0, np.sqrt(2.0 * gamma), size=(X.shape[1], self.n_components))
        self.offsets_ = rng.uniform(0.0, 2.0 * np.pi, size=self.n_components)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        check_is_fitted(self, "weights_")
        X = _as_2d(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        return np.sqrt(2.0 / self.n_components) * np.cos(X @ self.weights_ + self.offsets_)


class NystroemRBF(TransformerMixin, BaseEstimator):
    """Nyström approximation of the RBF kernel.

    Picks ``m`` landmark points and uses the low-rank approximation
    ``K ~= K_nm K_mm^{-1} K_mn``. The feature map is
    ``z(x) = K(x, landmarks) @ K_mm^{-1/2}``. Usually more accurate than Random
    Fourier Features for the same number of components because it adapts to the data.
    """

    def __init__(self, gamma: float = 1.0, n_components: int = 100, random_state=None):
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X: ArrayLike, y=None):
        X = _as_2d(X)
        _check_gamma(self.gamma)
        rng = np.random.default_rng(self.random_state)
        m = min(self.n_components, X.shape[0])
        idx = rng.choice(X.shape[0], size=m, replace=False)
        self.landmarks_ = X[idx]
        K_mm = rbf_kernel(self.landmarks_, gamma=self.gamma)
        # Symmetric inverse square root, discarding numerically-zero eigenvalues.
        eigval, eigvec = np.linalg.eigh(K_mm)
        keep = eigval > max(eigval.max(), 0) * 1e-12
        self.normalization_ = eigvec[:, keep] / np.sqrt(eigval[keep])
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        check_is_fitted(self, "landmarks_")
        return rbf_kernel(X, self.landmarks_, gamma=self.gamma) @ self.normalization_
