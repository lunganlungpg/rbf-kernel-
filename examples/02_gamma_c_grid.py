"""See over- and under-fitting: decision boundaries for a grid of gamma and C values.

Run: python examples/02_gamma_c_grid.py   (writes docs/images/gamma_c_grid.png)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from rbf_kernel import make_rbf_svc
from rbf_kernel.plotting import plot_decision_boundary

X, y = make_moons(n_samples=300, noise=0.25, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=0)

gammas = [0.1, 1.0, 10.0, 100.0]
Cs = [0.1, 1.0, 100.0]
fig, axes = plt.subplots(len(Cs), len(gammas), figsize=(12, 8.5))
print(f"{'':>8}" + "".join(f"gamma={g:<8}" for g in gammas))
for i, C in enumerate(Cs):
    line = f"C={C:<6}"
    for j, g in enumerate(gammas):
        model = make_rbf_svc(gamma=g, C=C).fit(X_tr, y_tr)
        tr, te = model.score(X_tr, y_tr), model.score(X_te, y_te)
        line += f"{tr:.2f}/{te:.2f}     "
        plot_decision_boundary(
            model, X_tr, y_tr, ax=axes[i, j], title=f"γ={g}, C={C}\ntrain {tr:.2f} · test {te:.2f}"
        )
    print(line)

fig.suptitle("RBF SVM on two moons (train / test accuracy)", fontsize=13)
fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "docs" / "images" / "gamma_c_grid.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=90)
print(f"\nSaved {out}")
print("Low gamma underfits (straight-ish boundary); high gamma overfits (islands around points).")
