"""Reproducible case study: breast cancer diagnosis with a tuned RBF SVM.

Uses the Wisconsin Diagnostic Breast Cancer dataset bundled with scikit-learn
(569 samples, 30 features). Run: python examples/03_breast_cancer.py
"""

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report, confusion_matrix, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from rbf_kernel import tune_rbf_svc

data = load_breast_cancer()
# In this dataset 0 = malignant, 1 = benign. Flip it so "positive" means malignant.
X, y = data.data, 1 - data.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

# Tune on ROC-AUC (threshold-free). Report sensitivity too: missing a malignant
# tumour is the costly error in this problem.
search = tune_rbf_svc(X_tr, y_tr, Cs=[0.1, 1, 10, 100], scoring="roc_auc")
model = search.best_estimator_
print(
    "Best params:", {k.split("__")[1]: round(float(v), 4) for k, v in search.best_params_.items()}
)
print(f"CV ROC-AUC: {search.best_score_:.4f}\n")

pred = model.predict(X_te)
print(classification_report(y_te, pred, target_names=["benign", "malignant"], digits=3))
print("Confusion matrix [rows = true benign, malignant]:\n", confusion_matrix(y_te, pred))
print(f"Test sensitivity (malignant recall): {recall_score(y_te, pred):.3f}")
print(f"Test ROC-AUC: {roc_auc_score(y_te, model.decision_function(X_te)):.4f}")
