# Changelog

## 1.0.0

- New `rbf_kernel` Python package: `rbf_kernel`, `pairwise_sq_dists`, `gamma_scale`, `median_heuristic_gamma`,
  `sigma_to_gamma` / `gamma_to_sigma`, `KernelRidgeRBF`, `RandomFourierFeatures`, `NystroemRBF`,
  `make_rbf_svc`, `tune_rbf_svc`, `gamma_grid` and optional plotting helpers.
- Test suite checking results against scikit-learn.
- Five runnable examples, including reproducible breast cancer and digits case studies.
- Rebuilt website: kernel profile, draggable similarity map, live kernel classifier playground
  with leave-one-out accuracy, tuning guide, light/dark themes, mobile layout.
- Site moved to `site/`; case studies page cleaned up (removed broken `/_sdk/` and injected scripts).
- CI (lint, tests on Python 3.11–3.13, example smoke tests), MIT license, contributing guide, issue templates.
