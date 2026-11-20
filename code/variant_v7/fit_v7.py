"""Подбор параметров v4. Запуск: python fit_v4.py <seed> <n> [rl_delay: 0/1]"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reconstruct_v7 as R  # noqa: E402

T = {
    "GOAP": ((14.0, 6.2, 6.0, 26.0), (21.0, 6.5, 11.0, 32.0)),
    "RL": ((54.0, 6.0, 43.0, 64.0), (50.0, 6.0, 42.0, 60.0)),
    "RL+кластеры+GOAP": ((19.0, 6.5, 10.0, 31.0), (23.0, 6.5, 13.0, 33.0)),
}
W = (1.0, 0.5, 0.25, 0.25)


def make(params):
    p = dict(R.FIT)
    p.update(params)
    p["p_target"] = tuple(np.array([params.get("w0", 0.4), params.get("w1", 0.3), params.get("w2", 0.3)]) / (params.get("w0", 0.4) + params.get("w1", 0.3) + params.get("w2", 0.3)))
    return p


def evaluate(params):
    p = make(params)
    built = R.build(p)
    errs, rows = 0.0, {}
    acc = {n: ([], []) for n in T}
    for seed in range(1, 4):
        res = R.run_case(seed, built, p)
        for n in T:
            (cr, rg), _ = res[n]
            acc[n][0].append(cr)
            acc[n][1].append(rg)
    pc = dict(p)
    pc["lam_test"] = p["lam_train"]
    rc = R.run_case(1, built, pc)
    mr = float(np.mean(np.concatenate(rc["RL"][0])))
    mg = float(np.mean(np.concatenate(rc["GOAP"][0])))
    errs += 3.0 * max(0.0, (mr - 1.25 * mg) / mg) ** 2
    rows["контроль RL/GOAP"] = (round(mr, 1), round(mg, 1))
    for n in T:
        for k in (0, 1):
            x = np.concatenate(acc[n][k]) if len(acc[n][k][0]) else np.array([0.0])
            got = (x.mean(), x.std(), np.percentile(x, 5), np.percentile(x, 95))
            for g, t_, w in zip(got, T[n][k], W):
                errs += w * ((g - t_) / t_) ** 2
            rows[f"{n}:{'крит' if k == 0 else 'ост'}"] = tuple(round(float(g), 1) for g in got)
    return errs, params, rows


def sample(rng, delay):
    d = dict(mis=float(rng.uniform(0.0, 40.0)), bp=float(rng.uniform(0.2, 4.0)), wait_t=float(rng.uniform(0.5, 4.0)), scale=float(rng.uniform(0.8, 3.0)), kappa=float(rng.uniform(0.3, 3.0)), cap=float(rng.uniform(2.0, 6.0)), sigma=float(rng.uniform(0.1, 0.7)),
             yield_wait=float(rng.uniform(1.0, 8.0)), crit_share=float(rng.uniform(0.2, 0.55)), lam_train=float(rng.choice([0.3, 0.5, 0.8])),
             lam_test=float(rng.uniform(1.0, 3.0)), lb=int(rng.choice([1, 2, 3, 5])), quantile=float(rng.uniform(0.85, 0.999)),
             wander=float(rng.uniform(0.5, 1.0)), eps=float(rng.uniform(0.15, 0.5)), w0=float(rng.uniform(0.2, 0.6)), w1=float(rng.uniform(0.2, 0.5)), w2=float(rng.uniform(0.1, 0.5)))
    if delay:
        d["rl_delay"] = float(rng.uniform(0.5, 6.0))
    return d


def refine(top, rng, delay):
    b = dict(top[int(rng.integers(len(top)))][1])
    for k, lo, hi in (("scale", 0.9, 1.1), ("kappa", 0.8, 1.2), ("sigma", 0.8, 1.2), ("yield_wait", 0.8, 1.2), ("crit_share", 0.9, 1.1), ("lam_test", 0.9, 1.1), ("wander", 0.9, 1.1), ("bp", 0.8, 1.25), ("mis", 0.8, 1.25), ("wait_t", 0.8, 1.25)):
        b[k] = float(np.clip(b[k] * rng.uniform(lo, hi), 0.01, 3.5 if k != "wander" else 1.0))
    if delay:
        b["rl_delay"] = float(b.get("rl_delay", 2.0) * rng.uniform(0.8, 1.25))
    b["quantile"] = float(rng.uniform(0.85, 0.9995))
    return b


if __name__ == "__main__":
    seed, n = int(sys.argv[1]), int(sys.argv[2])
    delay = len(sys.argv) > 3 and sys.argv[3] == "1"
    rng = np.random.default_rng(seed)
    with ProcessPoolExecutor() as ex:
        out = sorted(ex.map(evaluate, [sample(rng, delay) for _ in range(n)]), key=lambda x: x[0])
        for _ in range(4):
            more = list(ex.map(evaluate, [refine(out[:4], rng, delay) for _ in range(n // 2)]))
            out = sorted(out + more, key=lambda x: x[0])
    e, prm, rows = out[0]
    tag = "delay" if delay else "plain"
    print(f"[v7 {tag}] ошибка {e:.3f}")
    for k, v in rows.items():
        print("   ", k, v)
    Path(__file__).with_name(f"_v7_{tag}.json").write_text(json.dumps([e, prm, rows], ensure_ascii=False), encoding="utf-8")
