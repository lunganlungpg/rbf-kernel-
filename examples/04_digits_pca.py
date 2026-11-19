"""Reproducible case study: handwritten digits with PCA + RBF SVM.

Uses scikit-learn's bundled 8x8 digits (1,797 images, 64 pixels), so it runs
offline in seconds. PCA keeps 95% of the variance, then an RBF SVM classifies.
Run: python examples/04_digits_pca.py
"""

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

pipe = Pipeline(
    [
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=0.95, random_state=0)),
        ("svc", SVC(kernel="rbf")),
    ]
)
search = GridSearchCV(
    pipe, {"svc__gamma": ["scale", 0.005, 0.01, 0.05], "svc__C": [1, 10, 100]}, cv=5, n_jobs=-1
)
search.fit(X_tr, y_tr)
n_comp = search.best_estimator_.named_steps["pca"].n_components_

print(f"PCA: {X.shape[1]} pixels -> {n_comp} components (95% variance)")
print("Best params:", search.best_params_)
print(f"CV accuracy:   {search.best_score_:.4f}")
print(f"Test accuracy: {accuracy_score(y_te, search.predict(X_te)):.4f}")
