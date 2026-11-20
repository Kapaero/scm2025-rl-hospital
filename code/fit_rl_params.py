"""Подбор потерянных параметров реконструкции по целям, снятым с рисунков статьи (SCM'2025, RL + кластеризация + GOAP).

Цели (средние время до начала осмотра, мин; оценены по рисункам с точностью около 2 мин):
  GOAP: критические 14, остальные 21;  чистый RL: критические 53, остальные 49;  гибрид RL+кластеры+GOAP: критические 19, остальные 23.
Случайный поиск по допустимым диапазонам; критерий - средняя квадратичная относительная ошибка средних по 4 начальным значениям.
"""
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reconstruct_rl_cluster as R  # noqa: E402

TARGET = {"GOAP": (14.0, 21.0), "RL": (53.0, 49.0), "RL+кластеры+GOAP": (19.0, 23.0)}
TARGET_SD = {"GOAP": (6.0, 6.5), "RL": (7.0, 7.0), "RL+кластеры+GOAP": (6.0, 6.5)}   # разброс оценён по рисункам, точность около 2 мин


def evaluate(params):
    p = dict(R.FIT)
    p.update(params)
    p["length"] = tuple(np.array(R.FIT["length"]) * params.get("len_scale", 1.0))
    p["headway"] = tuple(np.array(R.FIT["headway"]) * params.get("hw_scale", 1.0))
    rl, C, mean, sd, tau, ns = R.build(1000, p)
    errs, rows = [], {}
    for name in TARGET:
        cm, rm, cs, rs = [], [], [], []
        for seed in range(1, 5):
            res = R.run_case(seed, rl, C, mean, sd, tau, p)
            (cr, rg), fl = res[name]
            cm.append(cr.mean())
            rm.append(rg.mean())
            cs.append(cr.std())
            rs.append(rg.std())
        c, r = float(np.mean(cm)), float(np.mean(rm))
        sc_, sr_ = float(np.mean(cs)), float(np.mean(rs))
        rows[name] = (round(c, 1), round(r, 1), round(sc_, 1), round(sr_, 1))
        errs += [((c - TARGET[name][0]) / TARGET[name][0]) ** 2, ((r - TARGET[name][1]) / TARGET[name][1]) ** 2,
                 0.15 * ((sc_ - TARGET_SD[name][0]) / TARGET_SD[name][0]) ** 2, 0.15 * ((sr_ - TARGET_SD[name][1]) / TARGET_SD[name][1]) ** 2]
    return float(np.mean(errs)), params, rows


if __name__ == "__main__":
    rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 48
    cands = []
    for _ in range(n):
        cands.append(dict(
            lam_test=float(rng.uniform(1.15, 1.8)), lam_train=float(rng.choice([0.55, 0.8])), crit_share=float(rng.uniform(0.22, 0.55)),
            boxes=int(rng.choice([6, 8, 10])), hw_scale=float(rng.choice([0.7, 0.8, 1.0])), len_scale=float(rng.choice([2.5, 3.0, 3.5, 4.0])),
            quantile=float(rng.uniform(0.80, 0.99)), qbin=int(rng.choice([1, 2, 3])), noise=float(rng.uniform(0.0, 1.2))))
    with ProcessPoolExecutor() as ex:
        out = list(ex.map(evaluate, cands))
    out.sort(key=lambda x: x[0])
    for e, prm, rows in out[:6]:
        print(round(e, 4), json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in prm.items()}, ensure_ascii=False), rows)
    Path(__file__).with_name("_fit_top.json").write_text(json.dumps([(e, prm, rows) for e, prm, rows in out[:10]], ensure_ascii=False), encoding="utf-8")

