"""Scaling RBF models: exact SVC vs Random Fourier Features vs Nyström.

Exact kernel SVMs get slow beyond ~50k samples. Approximate feature maps let a
linear model do the work. Run: python examples/05_scaling_up.py
"""

import time

from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from rbf_kernel import NystroemRBF, RandomFourierFeatures

X, y = make_classification(
    n_samples=20_000, n_features=20, n_informative=10, n_clusters_per_class=4, random_state=0
)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
gamma = 1 / X.shape[1]

models = {
    "Exact RBF SVC": make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma=gamma)),
    "RFF (1000) + linear": make_pipeline(
        StandardScaler(),
        RandomFourierFeatures(gamma, 1000, random_state=0),
        SGDClassifier(loss="hinge", alpha=1e-5, random_state=0),
    ),
    "Nyström (500) + linear": make_pipeline(
        StandardScaler(),
        NystroemRBF(gamma, 500, random_state=0),
        SGDClassifier(loss="hinge", alpha=1e-5, random_state=0),
    ),
}
print(f"{'model':<24}{'fit (s)':>9}{'test acc':>10}")
for name, model in models.items():
    t = time.perf_counter()
    model.fit(X_tr, y_tr)
    print(f"{name:<24}{time.perf_counter() - t:>9.2f}{model.score(X_te, y_te):>10.4f}")
