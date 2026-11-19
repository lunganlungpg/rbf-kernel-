"""What does the RBF kernel actually compute? Run: python examples/01_kernel_basics.py"""

import numpy as np

from rbf_kernel import gamma_to_sigma, rbf_kernel

points = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
origin = points[:1]

print("Similarity of each point to the origin, K(origin, x):\n")
print(f"{'distance':>10}" + "".join(f"  gamma={g:<5}" for g in (0.1, 1.0, 10.0)))
for p in points:
    row = [rbf_kernel(origin, p[None, :], gamma=g)[0, 0] for g in (0.1, 1.0, 10.0)]
    print(f"{p[0]:>10.1f}" + "".join(f"  {k:<11.4f}" for k in row))

print("\nLarger gamma means a narrower kernel (smaller length scale sigma):")
for g in (0.1, 1.0, 10.0):
    print(f"  gamma={g:<5} -> sigma = {gamma_to_sigma(g):.3f}")

K = rbf_kernel(points, gamma=1.0)
print("\nGram matrix (gamma=1):\n", np.round(K, 3))
print("Symmetric:", np.allclose(K, K.T), "| diagonal all 1:", np.allclose(np.diag(K), 1))
print(
    "Smallest eigenvalue (>= 0 means positive semi-definite):", f"{np.linalg.eigvalsh(K).min():.2e}"
)
