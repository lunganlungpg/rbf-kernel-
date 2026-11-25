# rbfkernel: RBF kernel methods for R

The RBF (Gaussian) kernel, kernel ridge regression and classification, random
Fourier features, the Nystroem approximation and RBF SVMs, with **a one-call
workflow that models any dataset**. The R companion to the
[RBF Kernel Explorer](https://lunganlungpg.github.io/rbf-kernel-/).

## Install

```r
# install.packages("remotes")
remotes::install_github("lunganlungpg/rbf-kernel-", subdir = "R-package", build_vignettes = TRUE)
```

Only base R is required; `e1071` enables the SVM option.

## Model any dataset in one call

```r
library(rbfkernel)
wf <- rbf_workflow(breast_cancer, target = "diagnosis", output_dir = "results")
wf                                 # test accuracy, baseline, confusion matrix, notes
plot(wf)                           # tuning heat map, confusion matrix (+ decision regions in 2-D)
predict(wf, new_patients)          # same encoding and scaling applied automatically
```

`rbf_workflow()` uses every other column as a feature (one-hot encoding
categorical ones), detects classification or regression, makes a stratified
train/test split, standardises on the training rows, tunes `gamma` and
`lambda` by **exact leave-one-out** (or an SVM by k-fold CV with
`method = "svm"`), evaluates on the held-out rows against a baseline, and
writes CSV tables, a PNG figure and a report when `output_dir` is set.

| Data | Call | Test result |
|---|---|---|
| Breast cancer (569 x 30) | `rbf_workflow(breast_cancer, "diagnosis")` | 97.2% accuracy (baseline 62.7%) |
| Iris (3 classes) | `rbf_workflow(iris, "Species")` | 94.4% accuracy |
| Two moons | `rbf_workflow(make_moons(400, 0.25), "class")` | 91.0% accuracy |
| Concentric circles | `rbf_workflow(make_circles(400, 0.08), "class")` | 100% accuracy |
| mtcars mpg (regression) | `rbf_workflow(mtcars, "mpg")` | RMSE 1.95 (baseline 3.78; only 8 test rows) |

## Building blocks

| Function | Purpose |
|---|---|
| `rbf_kernel()`, `sq_dist()` | Kernel and distance matrices |
| `gamma_scale()`, `gamma_median()`, `sigma_to_gamma()`, `gamma_to_sigma()` | Bandwidth heuristics and conversions |
| `krr()` + `predict()` | Kernel ridge regression / classification (scores or classes) |
| `krr_tune()` | Exact leave-one-out tuning over a gamma x lambda grid |
| `svm_tune()` | RBF SVM tuned by k-fold CV (needs e1071) |
| `rff_map()`, `nystroem_map()` | Explicit feature maps for large data |
| `make_moons()`, `make_circles()`, `plot_decision_boundary()` | Test problems and 2-D plots |

## Why the tuning is fast

For kernel ridge, the leave-one-out residual is
`alpha_i / [(K + lambda I)^-1]_ii`. One eigendecomposition per `gamma` gives
exact leave-one-out scores for every `lambda` without refitting. The test
suite checks this against brute-force refitting.

See `vignette("rbfkernel")`.
