"""Optional matplotlib helpers (``pip install rbf-kernel[plot]``)."""

from __future__ import annotations

import numpy as np


def plot_decision_boundary(model, X, y, ax=None, resolution: int = 300, title: str | None = None):
    """Draw a fitted 2-D classifier's decision function with the training points."""
    import matplotlib.pyplot as plt

    X = np.asarray(X)
    ax = ax or plt.gca()
    pad = 0.5
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, resolution),
        np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, resolution),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    if hasattr(model, "decision_function"):
        zz = model.decision_function(grid)
    else:
        zz = model.predict(grid)
    zz = zz.reshape(xx.shape)
    lim = np.percentile(np.abs(zz), 98) or 1.0
    ax.contourf(xx, yy, zz, levels=21, cmap="RdBu", vmin=-lim, vmax=lim, alpha=0.75)
    ax.contour(xx, yy, zz, levels=[0], colors="k", linewidths=1.5)
    y = np.asarray(y)
    palette = ["#b2182b", "#2166ac", "#1b7837", "#762a83"]
    for i, cls in enumerate(np.unique(y)):
        m = y == cls
        ax.scatter(X[m, 0], X[m, 1], c=palette[i % 4], edgecolors="white", linewidths=0.6, s=20)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)
    return ax
