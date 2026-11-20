"""Illustration of the clustering step.
State vector = occupancy of the nine corridors (edges) at the moment of decision; PCA, k-means (k=5);
the test-load state farthest from the centres is marked.
This is variant v7 of the model. It is used only for this picture; its waiting times are not reported
(they were not stable across seeds).
Usage: python cluster_figure.py <output_dir>
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_v7 as F  # noqa: E402
import reconstruct_v7 as R  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
OUT.mkdir(parents=True, exist_ok=True)
e, prm, _ = json.loads(Path(__file__).with_name("_v7_plain.json").read_text(encoding="utf-8"))
p = F.make(prm)
E = list(R.EDGES)


def collect(lam, seed, eps):
    rng = np.random.default_rng(seed)
    g = R.Goap(eps, rng)
    X = []

    class Rec:
        avoid = False

        def choose(self, a, state, feat, sim):
            X.append([sim.on_edge.get(k, 0) for k in E] + [feat[3]])
            return g.choose(a, state, feat, sim)

    R.Sim(lam, seed, Rec(), p).run()
    return np.array(X, float)


tr = np.vstack([collect(p["lam_train"], 1000 + s, p["eps"]) for s in range(20)])
te = collect(p["lam_test"], 1, 0.0)
mu, sd = tr.mean(0), tr.std(0) + 1e-9
Z, Zt = (tr - mu) / sd, (te - mu) / sd
U, S, Vt = np.linalg.svd(Z, full_matrices=False)
P, Pt = Z @ Vt[:2].T, Zt @ Vt[:2].T
lo, hi = P.min(0), P.max(0)
P, Pt = (P - lo) / (hi - lo), (Pt - lo) / (hi - lo)
rng = np.random.default_rng(3)
C = R.kmeans(P, 5, rng)
lab = np.argmin(((P[:, None] - C[None]) ** 2).sum(-1), 1)
dtr = np.min(np.linalg.norm(P[:, None] - C[None], axis=2), 1)
tau = np.quantile(dtr, 0.99)
dte = np.min(np.linalg.norm(Pt[:, None] - C[None], axis=2), 1)
far = int(np.argmax(dte))
print("explained variance PC1, PC2:", np.round((S ** 2 / (S ** 2).sum())[:2], 3),
      "share of unfamiliar test states:", round(float((dte > tau).mean()), 3),
      "threshold", round(float(tau), 3), "max distance", round(float(dte[far]), 3))
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.4})
fig, ax = plt.subplots(1, 1, figsize=(8, 5.2))
idx = np.random.default_rng(0).choice(len(P), 250, replace=False)
for k in range(5):
    m = lab[idx] == k
    ax.scatter(P[idx][m, 0], P[idx][m, 1], s=22, label=f"Cluster {k}")
ax.scatter(C[:, 0], C[:, 1], marker="X", s=200, c="k", label="Centres")
verdict = "above" if dte[far] > tau else "below"
ax.scatter(*np.clip(Pt[far], -0.05, 1.05), marker="*", s=180, c="r",
           label=f"Farthest test-load state (distance {dte[far]:.2f}, {verdict} threshold {tau:.2f})")
ax.set_title("Model variant v7: occupancy of 9 corridors at the moment of decision, PCA and k-means")
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
fig.tight_layout()
fig.savefig(OUT / "clusters_v7_illustration.png", dpi=120, bbox_inches="tight")
