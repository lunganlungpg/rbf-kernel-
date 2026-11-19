# Applied RBF case studies

The [case studies page](https://lunganlungpg.github.io/rbf-kernel-/case-studies.html) walks through how an RBF model is typically set up in five domains: the problem, why a Gaussian kernel fits it, a configuration, and the trade-offs.

| # | Domain | Task | Typical setup |
|---|---|---|---|
| 1 | Healthcare | Breast tumour classification (malignant vs benign) | Scaled RBF SVM tuned for sensitivity |
| 2 | Biometrics | Face verification | RBF on face embeddings |
| 3 | Cybersecurity | Network intrusion detection | Strongly regularized fit (large C) on traffic features |
| 4 | Finance | Volatility regime classification | RBF SVM on engineered market features |
| 5 | OCR | Handwritten digit recognition | PCA (784 → 154 components on MNIST) then RBF SVM |

**About the numbers.** The headline metrics on the page are illustrative and depend on the dataset, splits and tuning. Two scenarios can be reproduced end to end with this repository:

| Scenario | Script | Measured result |
|---|---|---|
| Breast cancer | [`examples/03_breast_cancer.py`](../examples/03_breast_cancer.py) | Test ROC-AUC 0.994, sensitivity 92.5% |
| Digits + PCA | [`examples/04_digits_pca.py`](../examples/04_digits_pca.py) | Test accuracy 98.4% (8×8 digits, 64 → 39 components) |

Contributions that add reproducible scripts for the other three scenarios on public datasets are very welcome.
