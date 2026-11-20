"""Реконструкция эксперимента статьи «Обучение с подкреплением для оптимизации распределения пациентов внутри больницы при ЧС» (SCM'2025).

Исходный код утрачен, поэтому модель восстановлена по описанию статьи, недостающие параметры подобраны (см. FIT).
Это РЕКОНСТРУКЦИЯ: совпадение с рисунками статьи достигнуто подбором параметров и не служит независимым подтверждением.

Блок: приёмное отделение с несколькими однополосными коридорами (маркеры-маршруты) и боксами. Пациенты двух классов (критические, остальные).
- GOAP: выбирает коридор по ожидаемому времени, критические имеют приоритет в очередях (уступают дорогу).
- RL: табличное Q-обучение по выборке случаев, записанных в симуляции GOAP; на незнакомом состоянии берёт действие по умолчанию.
- Гибрид: k-средних по состояниям выборки; если состояние далеко от центров (вне порога), решение принимает GOAP, иначе RL.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field

import numpy as np

FIT = dict(
    n_corr=3,
    length=(4.0, 6.0, 8.0),          # время прохода коридора, мин
    headway=(1.0, 1.5, 2.0),         # интервал между входами, мин
    boxes=8,
    treat=(2.0, 5.0),                # длительность контакта, мин
    crit_share=0.45,
    lam_train=0.55,                  # интенсивность поступления в обучающей выборке, 1/мин
    lam_test=1.55,                   # в проверяемом случае
    horizon=200.0,                   # длительность поступления, мин
    dt=0.25,
    qbin=2,                          # ширина ячейки длины очереди для таблицы Q
    kmeans_k=5,
    noise=0.0,                       # логарифмический шум времени прохода коридора
    quantile=0.95,
)


@dataclass
class Agent:
    t_arr: float
    crit: bool
    prio: bool = False
    corridor: int = -1
    t_enter: float = -1.0
    t_exit: float = -1.0
    t_treat: float = -1.0
    treat_len: float = 3.0
    jit: float = 1.0
    state: tuple = ()
    action: int = -1


class Sim:
    def __init__(self, lam: float, seed: int, policy, p=FIT):
        self.p = p
        self.rng = np.random.default_rng(seed)
        n = int(self.rng.poisson(lam * p["horizon"]))
        times = np.sort(self.rng.uniform(0, p["horizon"], n))
        self.agents = [Agent(float(t), bool(self.rng.random() < p["crit_share"]), treat_len=float(self.rng.uniform(*p["treat"])), jit=float(self.rng.lognormal(0.0, p.get("noise", 0.0)))) for t in times]
        self.policy = policy

    def run(self):
        p = self.p
        nc = p["n_corr"]
        queues = [[] for _ in range(nc)]
        next_free = [0.0] * nc
        walking = []                       # (t_exit, agent)
        box_free = [0.0] * p["boxes"]
        box_q = []
        t, i, n = 0.0, 0, len(self.agents)
        done = 0
        while done < n and t < p["horizon"] * 6:
            while i < n and self.agents[i].t_arr <= t:
                a = self.agents[i]
                i += 1
                qlen = [len(q) for q in queues]
                n_sys = sum(qlen) + len(walking) + len(box_q)
                a.state = (a.crit, tuple(min(q // p["qbin"], 50) for q in qlen))
                feat = np.array([float(a.crit)] + [float(q) for q in qlen] + [float(n_sys)])
                c, prio = self.policy.choose(a, queues, qlen, feat, self)
                a.corridor, a.action, a.prio = c, c, prio
                queues[c].append(a)
            for c in range(nc):
                if queues[c] and t >= next_free[c]:
                    q = queues[c]
                    k = 0
                    cand = [j for j, x in enumerate(q) if x.prio]
                    if cand:
                        k = max(cand, key=lambda j: (q[j].crit, -q[j].t_arr))
                    a = q.pop(k)
                    a.t_enter = t
                    a.t_exit = t + p["length"][c] * a.jit
                    walking.append(a)
                    next_free[c] = t + p["headway"][c]
            for a in [x for x in walking if x.t_exit <= t]:
                walking.remove(a)
                box_q.append(a)
            if box_q:
                for b in range(p["boxes"]):
                    if box_free[b] <= t and box_q:
                        cand = [j for j, x in enumerate(box_q) if x.prio and x.crit]
                        k = cand[0] if cand else 0
                        a = box_q.pop(k)
                        a.t_treat = t
                        box_free[b] = t + a.treat_len
                        done += 1
            t += p["dt"]
        return self.agents


class GoapPolicy:
    """Выбор коридора по ожидаемому времени; критические уступают дорогу не обязаны и идут первыми."""

    def __init__(self, eps=0.0, rng=None):
        self.eps, self.rng = eps, rng

    def choose(self, a, queues, qlen, feat, sim):
        p = sim.p
        if self.eps and self.rng.random() < self.eps:
            return int(self.rng.integers(p["n_corr"])), True
        best, bc = 1e18, 0
        for c in range(p["n_corr"]):
            ahead = sum(1 for x in queues[c] if (x.prio and (x.crit or not a.crit)) or (not a.crit and True)) if not a.crit else sum(1 for x in queues[c] if x.crit)
            est = ahead * p["headway"][c] + p["length"][c]
            if est < best:
                best, bc = est, c
        return bc, True


class TabularRL:
    def __init__(self, p=FIT):
        self.p = p
        self.sum = {}
        self.cnt = {}

    def fit(self, agents):
        for a in agents:
            if a.t_treat < 0:
                continue
            key = (a.state, a.action)
            self.sum[key] = self.sum.get(key, 0.0) + (a.t_treat - a.t_arr)
            self.cnt[key] = self.cnt.get(key, 0) + 1

    def q(self, state, action):
        k = (state, action)
        return self.sum[k] / self.cnt[k] if k in self.cnt else None

    def choose(self, a, queues, qlen, feat, sim):
        nc = self.p["n_corr"]
        vals = [self.q(a.state, c) for c in range(nc)]
        known = [(v, c) for c, v in enumerate(vals) if v is not None]
        if not known:
            return 0, False                   # незнакомое состояние: действие по умолчанию
        return min(known)[1], False           # RL не уступает дороги: обслуживание в порядке очереди

    def seen(self, state):
        return any((state, c) in self.cnt for c in range(self.p["n_corr"]))


class HybridPolicy:
    def __init__(self, rl: TabularRL, goap: GoapPolicy, centers, mean, sd, tau):
        self.rl, self.goap, self.centers, self.mean, self.sd, self.tau = rl, goap, centers, mean, sd, tau
        self.flagged = 0
        self.total = 0

    def choose(self, a, queues, qlen, feat, sim):
        z = (feat - self.mean) / self.sd
        d = float(np.min(np.linalg.norm(self.centers - z, axis=1)))
        self.total += 1
        if d > self.tau:
            self.flagged += 1
            return self.goap.choose(a, queues, qlen, feat, sim)
        return self.rl.choose(a, queues, qlen, feat, sim)


def kmeans(X, k, rng, iters=60):
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        newC = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    return C


def build(seed_train=1000, p=FIT):
    rng = np.random.default_rng(seed_train)
    sample = []
    feats = []
    for e in range(25):                       # выборка: симуляции GOAP с исследованием действий
        pol = GoapPolicy(eps=0.35, rng=rng)

        class Rec:                            # запись признаков состояния
            def __init__(self):
                self.pol = pol

            def choose(self, a, queues, qlen, feat, sim):
                feats.append(feat)
                return self.pol.choose(a, queues, qlen, feat, sim)

        s = Sim(p["lam_train"], seed_train + e, Rec(), p)
        sample += s.run()
    rl = TabularRL(p)
    rl.fit(sample)
    X = np.array(feats)
    mean, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mean) / sd
    C = kmeans(Z, p["kmeans_k"], rng)
    d = np.min(np.linalg.norm(Z[:, None, :] - C[None], axis=2), axis=1)
    tau = float(np.quantile(d, p["quantile"]))
    return rl, C, mean, sd, tau, len(sample)


def stats(agents):
    cr = np.array([a.t_treat - a.t_arr for a in agents if a.crit and a.t_treat >= 0])
    rg = np.array([a.t_treat - a.t_arr for a in agents if (not a.crit) and a.t_treat >= 0])
    return cr, rg


def run_case(seed, rl, C, mean, sd, tau, p=FIT):
    out = {}
    for name in ("GOAP", "RL", "RL+кластеры+GOAP"):
        if name == "GOAP":
            pol = GoapPolicy()
        elif name == "RL":
            pol = rl
        else:
            pol = HybridPolicy(rl, GoapPolicy(), C, mean, sd, tau)
        ag = Sim(p["lam_test"], seed, pol, p).run()
        out[name] = (stats(ag), getattr(pol, "flagged", 0) / max(1, getattr(pol, "total", 1)))
    return out


if __name__ == "__main__":
    rl, C, mean, sd, tau, ns = build()
    print(f"выборка: {ns} случаев; ячеек Q: {len(rl.cnt)}; порог кластеров {tau:.2f}")
    res = run_case(1, rl, C, mean, sd, tau)
    for k, ((cr, rg), fl) in res.items():
        print(f"{k:20s} критические {cr.mean():5.1f}±{cr.std():4.1f} (n={len(cr)})  остальные {rg.mean():5.1f}±{rg.std():4.1f} (n={len(rg)})  вне контура {fl:.2f}")
