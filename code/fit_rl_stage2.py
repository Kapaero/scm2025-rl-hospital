"""Второй этап подбора: окрестность лучших наборов первого этапа, квантиль порога кластеров в широком диапазоне."""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_rl_params as F  # noqa: E402

if __name__ == "__main__":
    top = json.loads(Path(__file__).with_name("_fit_top.json").read_text(encoding="utf-8"))[:4]
    rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    cands = []
    for _ in range(n):
        base = top[int(rng.integers(len(top)))][1]
        c = dict(base)
        c["lam_test"] = float(base["lam_test"] * rng.uniform(0.9, 1.1))
        c["crit_share"] = float(np.clip(base["crit_share"] * rng.uniform(0.85, 1.15), 0.2, 0.8))
        c["quantile"] = float(rng.uniform(0.9, 0.9995))
        c["qbin"] = int(rng.choice([1, 2, 3]))
        c["noise"] = float(np.clip(base.get("noise", 0.3) * rng.uniform(0.7, 1.3), 0.0, 1.5))
        c["len_scale"] = float(base["len_scale"] * rng.uniform(0.9, 1.1))
        cands.append(c)
    with ProcessPoolExecutor() as ex:
        out = list(ex.map(F.evaluate, cands))
    out.sort(key=lambda x: x[0])
    for e, prm, rows in out[:5]:
        print(round(e, 4), json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in prm.items()}, ensure_ascii=False), rows)
    Path(__file__).with_name("_fit_best.json").write_text(json.dumps(out[0][1], ensure_ascii=False), encoding="utf-8")
