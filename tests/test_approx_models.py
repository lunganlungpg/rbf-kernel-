import numpy as np
import pytest
from sklearn.datasets import make_moons
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rbf_kernel import (
    KernelRidgeRBF,
    NystroemRBF,
    RandomFourierFeatures,
    gamma_grid,
    make_rbf_svc,
    rbf_kernel,
    tune_rbf_svc,
)

rng = np.random.default_rng(0)
X = rng.normal(size=(300, 3))


def _max_err(Z):
    return np.abs(Z @ Z.T - rbf_kernel(X, gamma=0.5)).max()


def test_rff_error_decreases_with_components():
    small = _max_err(RandomFourierFeatures(0.5, 50, random_state=0).fit_transform(X))
    large = _max_err(RandomFourierFeatures(0.5, 5000, random_state=0).fit_transform(X))
    assert large < small
    assert large < 0.1


def test_nystroem_exact_when_all_points_are_landmarks():
    Z = NystroemRBF(0.5, n_components=len(X), random_state=0).fit_transform(X)
    assert _max_err(Z) < 1e-6


def test_rff_reproducible_and_validates():
    a = RandomFourierFeatures(random_state=1).fit_transform(X)
    b = RandomFourierFeatures(random_state=1).fit_transform(X)
    np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError):
        RandomFourierFeatures().fit(X).transform(X[:, :2])


def test_kernel_ridge_matches_sklearn():
    y = np.sin(X[:, 0]) + 0.1 * rng.normal(size=len(X))
    ours = KernelRidgeRBF(gamma=0.4, alpha=0.5).fit(X, y).predict(X[:20])
    theirs = KernelRidge(kernel="rbf", gamma=0.4, alpha=0.5).fit(X, y).predict(X[:20])
    np.testing.assert_allclose(ours, theirs, atol=1e-8)


def test_kernel_ridge_scale_gamma():
    model = KernelRidgeRBF().fit(X, X[:, 0])
    assert model.gamma_ == pytest.approx(1 / (3 * X.var()))
    assert model.score(X, X[:, 0]) > 0.9


def test_rff_pipeline_solves_moons():
    Xm, ym = make_moons(500, noise=0.15, random_state=0)
    clf = make_pipeline(
        StandardScaler(), RandomFourierFeatures(1.0, 300, random_state=0), LogisticRegression()
    )
    assert clf.fit(Xm, ym).score(Xm, ym) > 0.95


def test_gamma_grid_centre():
    g = gamma_grid(np.zeros((5, 4)), n=5)
    assert g[2] == pytest.approx(0.25)
    assert np.all(np.diff(g) > 0)


def test_tune_rbf_svc():
    Xm, ym = make_moons(200, noise=0.2, random_state=0)
    search = tune_rbf_svc(Xm, ym, Cs=[1, 10], cv=3)
    assert search.best_score_ > 0.9
    assert set(search.best_params_) == {"svc__gamma", "svc__C"}
    assert make_rbf_svc().fit(Xm, ym).score(Xm, ym) > 0.9
