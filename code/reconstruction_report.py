"""Report: single case study (as in the paper), check on other seeds, control regime, figures.
Usage: python reconstruction_report.py <output_dir>
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reconstruct_rl_cluster as R  # noqa: E402

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})

best = json.loads(Path(__file__).with_name("_fit_best.json").read_text(encoding="utf-8"))
p = dict(R.FIT)
p.update(best)
p["length"] = tuple(np.array(R.FIT["length"]) * best.get("len_scale", 1.0))
p["headway"] = tuple(np.array(R.FIT["headway"]) * best.get("hw_scale", 1.0))
rl, C, mean, sd, tau, ns = R.build(1000, p)
NAMES = ("GOAP", "RL", "RL+кластеры+GOAP")
LABEL = {"GOAP": "GOAP", "RL": "Pure RL", "RL+кластеры+GOAP": "RL + clustering + GOAP"}

# --- кейс-стади (один прогон, seed 1)
res = R.run_case(1, rl, C, mean, sd, tau, p)
fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), sharey=False)
case_rows = []
for ax, name in zip(axs, NAMES):
    (cr, rg), fl = res[name]
    bins = np.arange(0, max(cr.max(), rg.max()) + 3, 2.0)
    ax.hist(rg, bins=bins, color="#3b5bdb", alpha=0.6, label="Non-critical")
    ax.hist(cr, bins=bins, color="#e03131", alpha=0.6, label="Critical")
    ax.set_title(LABEL[name])
    ax.set_xlabel("Time to start of examination, min")
    case_rows.append((name, cr.mean(), cr.std(), len(cr), rg.mean(), rg.std(), len(rg), fl))
axs[0].set_ylabel("Number of patients")
axs[0].legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "rl_case_hist.png", dpi=200)
plt.close(fig)

# --- проверка на других сидах и контроль
def many(lam, seeds):
    q = dict(p)
    q["lam_test"] = lam
    agg = {n: [] for n in NAMES}
    fl = []
    for s in seeds:
        r = R.run_case(s, rl, C, mean, sd, tau, q)
        for n in NAMES:
            (cr, rg), f = r[n]
            agg[n].append((cr.mean(), rg.mean()))
        fl.append(r["RL+кластеры+GOAP"][1])
    return agg, float(np.mean(fl))


lines = ["# Re-implementation of the experiment (RL + clustering + GOAP)", "",
         "Calibrated parameters: " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in best.items()}, ensure_ascii=False), ""]
lines.append("## Case study (single run, seed 1; as in the paper)")
lines.append("| Method | Critical, min (mean ± SD, n) | Non-critical, min (mean ± SD, n) | Out of contour |")
lines.append("|---|---|---|---|")
for name, cm, cs, cn, rm, rs, rn, fl in case_rows:
    lines.append(f"| {LABEL[name]} | {cm:.1f} ± {cs:.1f} ({cn}) | {rm:.1f} ± {rs:.1f} ({rn}) | {fl:.2f} |")
lines.append("")
lines.append("Targets read from the paper figures (mean, min): GOAP 14 / 21; pure RL 53 / 49; hybrid 19 / 23 (critical / non-critical).")
lines.append("")
for title, lam in (("Check on 30 other seeds (101-130), same load as the case study", p["lam_test"]), ("Control: load as in the training sample", p["lam_train"])):
    agg, fl = many(lam, range(101, 131))
    lines.append(f"## {title}")
    lines.append("| Method | Critical, min (mean over seeds ± SD between seeds) | Non-critical, min |")
    lines.append("|---|---|---|")
    for n in NAMES:
        a = np.array(agg[n])
        lines.append(f"| {LABEL[n]} | {a[:, 0].mean():.1f} ± {a[:, 0].std(ddof=1):.1f} | {a[:, 1].mean():.1f} ± {a[:, 1].std(ddof=1):.1f} |")
    lines.append(f"Share of decisions out of contour (hybrid): {fl:.2f}")
    lines.append("")

# --- PCA: обучающие состояния, центры и новый случай
rng = np.random.default_rng(3)
rows = []


class Rec:
    def __init__(self):
        self.pol = R.GoapPolicy()

    def choose(self, a, queues, qlen, feat, sim):
        rows.append(feat)
        return self.pol.choose(a, queues, qlen, feat, sim)


train_feats = []
for e in range(4):
    rows.clear()
    R.Sim(p["lam_train"], 2000 + e, Rec(), p).run()
    train_feats += rows
Xtr = np.array(train_feats)
rows.clear()
R.Sim(p["lam_test"], 7, Rec(), p).run()
Xte = np.array(rows)
Ztr, Zte = (Xtr - mean) / sd, (Xte - mean) / sd
pca = PCA(2).fit(Ztr)
Ptr, Pte, Pc = pca.transform(Ztr), pca.transform(Zte), pca.transform(C)
lab = np.argmin(((Ztr[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
dte = np.min(np.linalg.norm(Zte[:, None, :] - C[None], axis=2), axis=1)
fig, ax = plt.subplots(figsize=(6.4, 4.6))
for j in range(len(C)):
    ax.scatter(Ptr[lab == j, 0], Ptr[lab == j, 1], s=8, alpha=0.5, label=f"Cluster {j}")
ax.scatter(Pc[:, 0], Pc[:, 1], marker="X", s=120, color="k", label="Centres")
out = dte > tau
ax.scatter(Pte[out, 0], Pte[out, 1], marker="*", s=70, color="#e03131", label="Out-of-contour cases")
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.legend(frameon=False, fontsize=8)
ax.set_title("State clustering: familiar and new cases")
fig.tight_layout()
fig.savefig(OUT / "rl_pca.png", dpi=200)
plt.close(fig)
lines.append(f"Share of test-regime states out of contour: {out.mean():.2f}; in the training regime it is 0.05 by construction of the threshold (quantile {p['quantile']:.3f}).")
(OUT / "reconstruction.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
