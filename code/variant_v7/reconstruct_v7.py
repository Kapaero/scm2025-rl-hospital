"""Реконструкция v7 (v6 + цена неверного поворота; RL уступает/требует приоритет только в знакомых состояниях).

Граф маркеров, одно ребро - один встречный проход. Все идут по очереди; при встрече лицом к лицу на ребре обычный уступает критическому
(у GOAP и у RL, RL повторяет приоритет из выборки GOAP; уступающий возвращается на прежний маркер).
Обход столкновений есть только у гибрида RL+кластеры+GOAP: перед входом в занятое встречным ребро он выбирает другой маршрут или ждёт.
"""
from __future__ import annotations

import heapq

import numpy as np

import reconstruct_v4 as V4
from reconstruct_v4 import EDGES, NB, Goap, Hybrid, TabularRL, dijkstra_next, kmeans, stats  # noqa: F401

FIT = dict(V4.FIT)
FIT.update(bp=0.5, wait_t=1.0, mis=0.0)


def dijkstra_excl(sim, a, excl):
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
            if (min(u, v), max(u, v)) == excl:
                continue
            nd = d + sim.est(u, v)
            if nd < dist.get(v, 1e18):
                dist[v], prev[v] = nd, u
                heapq.heappush(pq, (nd, v))
    if a.target not in dist:
        return None
    u = a.target
    while prev.get(u, a.node) != a.node:
        u = prev[u]
    return u


class RL7(TabularRL):
    def choose(self, a, state, feat, sim):
        known = [(self.sum[(state, v)] / self.cnt[(state, v)], v) for v in NB[a.node] if (state, v) in self.cnt]
        if known:
            return min(known)[1], True, 0.0
        nb = list(NB[a.node])
        if self.rng.random() < self.p["wander"]:
            return nb[int(self.rng.integers(len(nb)))], False, 0.0
        return dijkstra_next(sim, a), False, 0.0


class Sim(V4.Sim):
    def __init__(self, lam, seed, policy, p=FIT):
        super().__init__(lam, seed, policy, p)
        for a in self.agents:
            a.dest, a.t_start, a.turned, a.yok = -1, 0.0, False, True
        self.collisions = 0
        self.cb = None

    def run(self):
        p = self.p
        avoid = bool(getattr(self.policy, "avoid", False))
        active, i, n, done = [], 0, len(self.agents), 0
        occ = {e: [] for e in EDGES}
        t = 0.0
        while done < n and t < p["horizon"] * 5:
            while i < n and self.agents[i].t_arr <= t:
                a = self.agents[i]
                i += 1
                a.t_next = t
                active.append(a)
            n_sys = len(active)
            for a in list(active):
                if a.edge and t >= a.t_next:
                    e = a.edge
                    occ[e].remove(a)
                    self.on_edge[e] -= 1
                    a.edge = ()
                    if not a.turned:
                        a.node = a.dest
                    a.turned = False
                    if a.node == a.target:
                        a.t_done = t
                        active.remove(a)
                        done += 1
            for a in list(active):
                if a.edge or t < a.t_next:
                    continue
                loc = sum(self.on_edge.get(e, 0) for e in EDGES if a.node in e)
                state = (a.node, a.target, a.crit, min(n_sys // p["lb"], 40))
                feat = np.array([float(a.node), float(a.target), float(a.crit), float(n_sys), float(loc)])
                nxt, yok, _ = self.policy.choose(a, state, feat, self)
                a.yok = yok
                wrong = p['mis'] > 0 and nxt != dijkstra_next(self, a)
                e = (min(a.node, nxt), max(a.node, nxt))
                opp = [x for x in occ[e] if x.dest == a.node and not x.turned]
                if opp:
                    if avoid:
                        alt = dijkstra_excl(self, a, e)
                        if alt is not None:
                            nxt = alt
                            e = (min(a.node, nxt), max(a.node, nxt))
                            opp = [x for x in occ[e] if x.dest == a.node and not x.turned]
                        if alt is None or opp:
                            a.t_next = t + p["wait_t"]
                            continue
                    else:
                        self.collisions += 1
                        if a.crit and a.yok and all((not x.crit) and x.yok for x in opp):
                            for x in opp:
                                x.turned = True
                                x.t_next = t + (t - x.t_start) + p["bp"]
                        elif (not a.crit) and a.yok and any(x.crit and x.yok for x in opp):
                            a.t_next = t + p["bp"]
                            continue
                        elif self.rng.random() < 0.5:
                            a.t_next = t + p["bp"]
                            continue
                        else:
                            for x in opp:
                                x.turned = True
                                x.t_next = t + (t - x.t_start) + p["bp"]
                a.log.append((state, nxt, t))
                a.edge, a.dest, a.t_start = e, nxt, t
                occ[e].append(a)
                self.on_edge[e] = self.on_edge.get(e, 0) + 1
                base = EDGES[e] * p["scale"] * (1.0 + p["kappa"] * (self.on_edge[e] - 1) / p["cap"])
                a.t_next = t + base * float(self.rng.lognormal(0.0, p["sigma"])) + (p['mis'] if wrong else 0.0)
            if self.cb:
                self.cb(t, active)
            t += p["dt"]
        return self.agents


def build(p=FIT, seed=1000):
    rng = np.random.default_rng(seed)
    sample, feats = [], []
    for e in range(20):
        g = Goap(p["eps"], rng)

        class Rec:
            avoid = False

            def choose(self, a, state, feat, sim, g=g):
                feats.append(feat)
                return g.choose(a, state, feat, sim)

        sample += Sim(p["lam_train"], seed + e, Rec(), p).run()
    rl = RL7(p)
    rl.fit(sample)
    X = np.array(feats)
    mean, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mean) / sd
    C = kmeans(Z, p["kmeans_k"], rng)
    d = np.min(np.linalg.norm(Z[:, None, :] - C[None], axis=2), axis=1)
    return rl, C, mean, sd, float(np.quantile(d, p["quantile"]))


def run_case(seed, built, p=FIT):
    rl, C, mean, sd, tau = built
    out = {}
    for name in ("GOAP", "RL", "RL+кластеры+GOAP"):
        if name == "GOAP":
            pol = Goap()
        elif name == "RL":
            pol = rl
        else:
            pol = Hybrid(rl, Goap(), C, mean, sd, tau)
            pol.avoid = True
        ag = Sim(p["lam_test"], seed, pol, p).run()
        out[name] = (stats(ag), getattr(pol, "flagged", 0) / max(1, getattr(pol, "total", 1)))
    return out
