"""rbf_kernel — a small, well-tested toolkit for understanding and using RBF kernels.

Everything here is implemented in plain NumPy so you can read the code and see
exactly how the Gaussian (RBF) kernel works. Where scikit-learn offers an
equivalent, the tests check that both give the same results.
"""

from .approx import NystroemRBF, RandomFourierFeatures
from .kernels import (
    gamma_scale,
    gamma_to_sigma,
    median_heuristic_gamma,
    pairwise_sq_dists,
    rbf_kernel,
    sigma_to_gamma,
)
from .models import KernelRidgeRBF
from .tuning import gamma_grid, make_rbf_svc, tune_rbf_svc

__all__ = [
    "KernelRidgeRBF",
    "NystroemRBF",
    "RandomFourierFeatures",
    "gamma_grid",
    "gamma_scale",
    "gamma_to_sigma",
    "make_rbf_svc",
    "median_heuristic_gamma",
    "pairwise_sq_dists",
    "rbf_kernel",
    "sigma_to_gamma",
    "tune_rbf_svc",
]

__version__ = "1.0.0"
