import numpy as np
import pytest
from sklearn.metrics.pairwise import rbf_kernel as sk_rbf

from rbf_kernel import (
    gamma_scale,
    gamma_to_sigma,
    median_heuristic_gamma,
    pairwise_sq_dists,
    rbf_kernel,
    sigma_to_gamma,
)

rng = np.random.default_rng(42)
X = rng.normal(size=(40, 5))
Y = rng.normal(size=(25, 5))


@pytest.mark.parametrize("gamma", [0.01, 0.2, 1.0, 10.0])
def test_matches_sklearn(gamma):
    np.testing.assert_allclose(rbf_kernel(X, Y, gamma), sk_rbf(X, Y, gamma=gamma), atol=1e-12)
    np.testing.assert_allclose(rbf_kernel(X, gamma=gamma), sk_rbf(X, gamma=gamma), atol=1e-12)


def test_gram_matrix_properties():
    K = rbf_kernel(X, gamma=0.3)
    np.testing.assert_allclose(np.diag(K), 1.0)  # self-similarity
    np.testing.assert_allclose(K, K.T)  # symmetric
    assert np.all((K > 0) & (K <= 1))  # bounded
    assert np.linalg.eigvalsh(K).min() > -1e-10  # positive semi-definite


def test_known_value():
    assert rbf_kernel([[0.0, 0.0]], [[1.0, 1.0]], gamma=0.5)[0, 0] == pytest.approx(np.exp(-1))


def test_one_dimensional_input_is_treated_as_column():
    assert rbf_kernel(np.arange(4.0), gamma=1).shape == (4, 4)


def test_sq_dists_non_negative_for_near_duplicates():
    Z = np.array([[1e8, 1e8], [1e8 + 1e-6, 1e8]])
    assert pairwise_sq_dists(Z).min() >= 0


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf])
def test_invalid_gamma(bad):
    with pytest.raises(ValueError):
        rbf_kernel(X, gamma=bad)


def test_feature_mismatch():
    with pytest.raises(ValueError, match="same number of features"):
        rbf_kernel(X, Y[:, :3])


def test_nan_rejected():
    Z = X.copy()
    Z[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        rbf_kernel(Z)


def test_sigma_gamma_roundtrip():
    for s in [0.1, 1.0, 3.7]:
        assert gamma_to_sigma(sigma_to_gamma(s)) == pytest.approx(s)
    assert sigma_to_gamma(1.0) == 0.5


def test_gamma_scale_matches_sklearn_definition():
    assert gamma_scale(X) == pytest.approx(1.0 / (X.shape[1] * X.var()))


def test_median_heuristic_scales_with_data():
    g1 = median_heuristic_gamma(X)
    g10 = median_heuristic_gamma(10 * X)
    assert g10 == pytest.approx(g1 / 100)


def test_median_heuristic_subsamples():
    big = rng.normal(size=(3000, 2))
    assert median_heuristic_gamma(big, max_samples=200) > 0
