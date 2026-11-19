"""Helpers for building and tuning RBF SVMs the right way."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def gamma_grid(X: ArrayLike, n: int = 7, decades: float = 3.0) -> np.ndarray:
    """A log-spaced gamma grid centred on ``1 / n_features``.

    Centre is the ``gamma='scale'`` value for *standardized* data, which is what
    :func:`make_rbf_svc` feeds the SVM. ``decades`` sets the total width, e.g.
    ``3`` spans 10^-1.5 to 10^1.5 times the centre.
    """
    X = np.asarray(X)
    n_features = 1 if X.ndim == 1 else X.shape[1]
    centre = 1.0 / n_features
    return centre * np.logspace(-decades / 2, decades / 2, n)


def make_rbf_svc(gamma: float | str = "scale", C: float = 1.0, **svc_kwargs) -> Pipeline:
    """``StandardScaler`` + RBF ``SVC`` in one pipeline.

    Putting the scaler *inside* the pipeline means cross-validation fits it on
    each training fold only, so there is no data leakage.
    """
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("svc", SVC(kernel="rbf", gamma=gamma, C=C, **svc_kwargs)),
        ]
    )


def tune_rbf_svc(
    X: ArrayLike,
    y: ArrayLike,
    gammas: ArrayLike | None = None,
    Cs: ArrayLike = (0.1, 1, 10, 100),
    cv: int = 5,
    scoring: str = "accuracy",
    n_jobs: int | None = None,
    random_state: int = 0,
) -> GridSearchCV:
    """Grid-search ``gamma`` and ``C`` for a scaled RBF SVM with stratified CV.

    Returns the fitted :class:`~sklearn.model_selection.GridSearchCV`; use
    ``.best_params_``, ``.best_score_`` and ``.best_estimator_``.
    """
    if gammas is None:
        gammas = gamma_grid(X)
    search = GridSearchCV(
        make_rbf_svc(),
        {"svc__gamma": list(gammas), "svc__C": list(Cs)},
        cv=StratifiedKFold(cv, shuffle=True, random_state=random_state),
        scoring=scoring,
        n_jobs=n_jobs,
    )
    return search.fit(X, y)
