"""A from-scratch kernel model so you can see the kernel trick end to end."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import cho_factor, cho_solve
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from .kernels import _as_2d, _check_gamma, gamma_scale, rbf_kernel


class KernelRidgeRBF(RegressorMixin, BaseEstimator):
    """Kernel ridge regression with an RBF kernel, in ~20 lines.

    Training solves ``(K + alpha * I) a = y`` and prediction is
    ``f(x) = sum_i a_i K(x, x_i)``. The model is a weighted sum of Gaussian
    "bumps" centred on the training points, which is the clearest way to see
    what an RBF kernel model actually computes.

    Parameters
    ----------
    gamma : float or "scale", default="scale"
    alpha : float, default=1.0
        Ridge regularization strength. Larger means smoother.
    """

    def __init__(self, gamma: float | str = "scale", alpha: float = 1.0):
        self.gamma = gamma
        self.alpha = alpha

    def fit(self, X: ArrayLike, y: ArrayLike):
        X = _as_2d(X)
        y = np.asarray(y, dtype=float)
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y have a different number of samples")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        self.gamma_ = gamma_scale(X) if self.gamma == "scale" else _check_gamma(self.gamma)
        K = rbf_kernel(X, gamma=self.gamma_)
        K[np.diag_indices_from(K)] += self.alpha
        self.dual_coef_ = cho_solve(cho_factor(K, lower=True), y)
        self.X_fit_ = X
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        check_is_fitted(self, "dual_coef_")
        return rbf_kernel(X, self.X_fit_, gamma=self.gamma_) @ self.dual_coef_
