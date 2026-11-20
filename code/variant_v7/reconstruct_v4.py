"""Реконструкция v4: пациент идёт по графу маркеров (отделений), время шага случайное и зависит от загрузки ребра.

- GOAP: следующий маркер по кратчайшему оценочному пути; обычные пациенты уступают дорогу критическим (ждут, пока на ребре нет критического).
- RL: табличная политика по выборке GOAP (ключ: маркер, цель, класс, ячейка загрузки); на незнакомом состоянии выбирает действие по режиму wander
  (случайный соседний маркер) и не уступает дорогу.
- Гибрид: k-средних по состояниям выборки; вне порога решает GOAP.
Время = от поступления до прибытия в целевой маркер.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

EDGES = {(0, 1): 4.0, (0, 2): 6.0, (1, 3): 5.0, (2, 3): 4.0, (1, 4): 7.0, (2, 5): 7.0, (3, 4): 5.0, (3, 5): 6.0, (4, 5): 4.0}
NB = {n: {} for n in range(6)}
for (a, b), w in EDGES.items():
    NB[a][b] = w
    NB[b][a] = w

FIT = dict(
    scale=1.0,                 # масштаб базовых времён шага
    kappa=1.0,                 # влияние загрузки ребра
    cap=3.0,
    sigma=0.35,                # логарифмический шум времени шага
    yield_wait=3.0,            # ожидание обычного пациента, если на ребре критический, мин
    targets=(3, 4, 5),
    p_target=(0.4, 0.3, 0.3),
    crit_share=0.35,
    lam_train=0.5,
    lam_test=1.6,
    horizon=200.0,
    dt=0.5,
    lb=2,                      # ширина ячейки загрузки для таблицы
    kmeans_k=5,
    quantile=0.95,
    wander=1.0,                # доля случайных решений RL на незнакомом состоянии
    eps=0.3,                   # исследование в обучающей выборке
    rl_delay=0.0,              # задержка решения RL, мин (колебание), опционально
)


@dataclass
class Agent:
    t_arr: float
    crit: bool
    target: int
    node: int = 0
    prev: int = -1
    t_next: float = 0.0
    edge: tuple = ()
    t_done: float = -1.0
    use_yield: bool = True
    log: list = field(default_factory=list)       # (состояние, действие, время решения)


class Sim:
    def __init__(self, lam, seed, policy, p=FIT):
        self.p = p
        self.rng = np.random.default_rng(seed)
        n = int(self.rng.poisson(lam * p["horizon"]))
        times = np.sort(self.rng.uniform(0, p["horizon"], n))
        self.agents = [Agent(float(t), bool(self.rng.random() < p["crit_share"]), int(self.rng.choice(p["targets"], p=p["p_target"]))) for t in times]
        self.policy = policy
        self.on_edge = {}
        self.crit_on_edge = {}

    def est(self, a, b):
        e = (min(a, b), max(a, b))
        return EDGES[e] * self.p["scale"] * (1.0 + self.p["kappa"] * self.on_edge.get(e, 0) / self.p["cap"])

    def run(self):
        p = self.p
        active, i, n, done = [], 0, len(self.agents), 0
        t = 0.0
        while done < n and t < p["horizon"] * 5:
            while i < n and self.agents[i].t_arr <= t:
                a = self.agents[i]
                i += 1
                a.t_next = t
                active.append(a)
            n_sys = len(active)
            for a in list(active):
                if a.edge:                                   # идёт по ребру
                    if t >= a.t_next:
                        e = a.edge
                        self.on_edge[e] -= 1
                        if a.crit:
                            self.crit_on_edge[e] -= 1
                        a.edge = ()
                        a.node = e[1] if a.node == e[0] else e[0]
                        if a.node == a.target:
                            a.t_done = t
                            active.remove(a)
                            done += 1
                    continue
                if t < a.t_next:
                    continue
                # решение о следующем маркере
                loc = sum(self.on_edge.get(e, 0) for e in EDGES if a.node in e)
                state = (a.node, a.target, a.crit, min(n_sys // p["lb"], 40))
                feat = np.array([float(a.node), float(a.target), float(a.crit), float(n_sys), float(loc)])
                nxt, use_yield, delay = self.policy.choose(a, state, feat, self)
                if delay > 0 and not getattr(a, "_delayed", False):
                    a._delayed = True
                    a.t_next = t + delay
                    continue
                a._delayed = False
                e = (min(a.node, nxt), max(a.node, nxt))
                if a.crit is False and use_yield and self.crit_on_edge.get(e, 0) > 0:
                    a.t_next = t + p["yield_wait"]            # уступает дорогу критическому
                    continue
                a.log.append((state, nxt, t))
                a.prev = a.node
                a.edge = e
                self.on_edge[e] = self.on_edge.get(e, 0) + 1
                if a.crit:
                    self.crit_on_edge[e] = self.crit_on_edge.get(e, 0) + 1
                base = EDGES[e] * p["scale"] * (1.0 + p["kappa"] * (self.on_edge[e] - 1) / p["cap"])
                a.t_next = t + base * float(self.rng.lognormal(0.0, p["sigma"]))
            t += p["dt"]
        return self.agents


def dijkstra_next(sim, a):
    dist = {a.node: 0.0}
    prev = {}
    pq = [(0.0, a.node)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == a.target:
            break
        if d > dist.get(u, 1e18):
            continue
        for v in NB[u]:
            nd = d + sim.est(u, v)
            if nd < dist.get(v, 1e18):
                dist[v], prev[v] = nd, u
                heapq.heappush(pq, (nd, v))
    u = a.target
    while prev.get(u, a.node) != a.node:
        u = prev[u]
    return u


class Goap:
    def __init__(self, eps=0.0, rng=None):
        self.eps, self.rng = eps, rng

    def choose(self, a, state, feat, sim):
        if self.eps and self.rng.random() < self.eps:
            nb = list(NB[a.node])
            return nb[int(self.rng.integers(len(nb)))], True, 0.0
        return dijkstra_next(sim, a), True, 0.0


class TabularRL:
    def __init__(self, p=FIT):
        self.p, self.sum, self.cnt = p, {}, {}
        self.rng = np.random.default_rng(5)

    def fit(self, agents):
        for a in agents:
            if a.t_done < 0:
                continue
            for state, act, td in a.log:
                k = (state, act)
                self.sum[k] = self.sum.get(k, 0.0) + (a.t_done - td)
                self.cnt[k] = self.cnt.get(k, 0) + 1

    def choose(self, a, state, feat, sim):
        known = [(self.sum[(state, v)] / self.cnt[(state, v)], v) for v in NB[a.node] if (state, v) in self.cnt]
        if known:
            return min(known)[1], False, self.p["rl_delay"]
        nb = list(NB[a.node])
        if self.rng.random() < self.p["wander"]:
            return nb[int(self.rng.integers(len(nb)))], False, self.p["rl_delay"]
        return dijkstra_next(sim, a), False, self.p["rl_delay"]


class Hybrid:
    def __init__(self, rl, goap, C, mean, sd, tau):
        self.rl, self.goap, self.C, self.mean, self.sd, self.tau = rl, goap, C, mean, sd, tau
        self.flagged = self.total = 0

    def choose(self, a, state, feat, sim):
        d = float(np.min(np.linalg.norm(self.C - (feat - self.mean) / self.sd, axis=1)))
        self.total += 1
        if d > self.tau:
            self.flagged += 1
            return self.goap.choose(a, state, feat, sim)
        return self.rl.choose(a, state, feat, sim)


def kmeans(X, k, rng):
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(60):
        lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        newC = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    return C


def build(p=FIT, seed=1000):
    rng = np.random.default_rng(seed)
    sample, feats = [], []
    for e in range(20):
        g = Goap(p["eps"], rng)

        class Rec:
            def choose(self, a, state, feat, sim, g=g):
                feats.append(feat)
                return g.choose(a, state, feat, sim)

        sample += Sim(p["lam_train"], seed + e, Rec(), p).run()
    rl = TabularRL(p)
    rl.fit(sample)
    X = np.array(feats)
    mean, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mean) / sd
    C = kmeans(Z, p["kmeans_k"], rng)
    d = np.min(np.linalg.norm(Z[:, None, :] - C[None], axis=2), axis=1)
    return rl, C, mean, sd, float(np.quantile(d, p["quantile"]))


def stats(agents):
    cr = np.array([a.t_done - a.t_arr for a in agents if a.crit and a.t_done >= 0])
    rg = np.array([a.t_done - a.t_arr for a in agents if (not a.crit) and a.t_done >= 0])
    return cr, rg


def run_case(seed, built, p=FIT):
    rl, C, mean, sd, tau = built
    out = {}
    for name in ("GOAP", "RL", "RL+кластеры+GOAP"):
        pol = Goap() if name == "GOAP" else (rl if name == "RL" else Hybrid(rl, Goap(), C, mean, sd, tau))
        ag = Sim(p["lam_test"], seed, pol, p).run()
        out[name] = (stats(ag), getattr(pol, "flagged", 0) / max(1, getattr(pol, "total", 1)))
    return out
