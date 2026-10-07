"""Where the construction does NOT help: market split (m equality constraints on random integers, planted solution).
Exact look-ahead means knowing which partial-sum vectors can still reach b, i.e. solving the instance.  The practical automaton is a
sound relaxation: state = vector of partial sums, forcing only when a bound makes a value impossible (sum would exceed b_i, or the
remaining items cannot reach b_i).  We report its width (distinct partial-sum vectors), the probability that A_{1/2}|0> ends feasible
(dead ends are lost mass) and the amplification iterations against Grover.  Writes results/general_limits.json."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)
import json, math
import numpy as np


def instance(n, m, seed, amax=99):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, amax + 1, (m, n))
    xs = rng.integers(0, 2, n)
    return a, a @ xs


def relaxed_walk(a, b):
    m, n = a.shape
    rem = np.concatenate([np.cumsum(a[:, ::-1], axis=1)[:, ::-1], np.zeros((m, 1), int)], axis=1)   # rem[:, k] = sum_{j>=k}
    dist = {tuple([0] * m): 1.0}
    width = []
    for k in range(n):
        nd = {}
        for s, p in dist.items():
            s = np.array(s)
            can0 = bool(np.all(s + rem[:, k + 1] >= b))             # still able to reach b with x_k = 0
            s1 = s + a[:, k]
            can1 = bool(np.all(s1 <= b)) and bool(np.all(s1 + rem[:, k + 1] >= b))
            if can0 and can1:
                nd[tuple(s)] = nd.get(tuple(s), 0) + p / 2
                nd[tuple(s1)] = nd.get(tuple(s1), 0) + p / 2
            elif can0:
                nd[tuple(s)] = nd.get(tuple(s), 0) + p
            elif can1:
                nd[tuple(s1)] = nd.get(tuple(s1), 0) + p
            # else: dead end, mass lost
        dist = nd; width.append(len(dist))
    return dist.get(tuple(b), 0.0), max(width)


def count_solutions(a, b):
    m, n = a.shape
    cnt = {tuple([0] * m): 1}
    for k in range(n):
        nc = {}
        for s, c in cnt.items():
            for bit in (0, 1):
                t = tuple(np.array(s) + bit * a[:, k])
                if all(t[i] <= b[i] for i in range(m)):
                    nc[t] = nc.get(t, 0) + c
        cnt = nc
    return cnt.get(tuple(b), 0)


def main():
    rows = []
    for m in (2, 3, 4):
        for n in (10, 12, 14, 16, 18, 20):
            res = []
            for seed in range(3):
                a, b = instance(n, m, 100 * n + 10 * m + seed)
                pf, w = relaxed_walk(a, b)
                nsol = count_solutions(a, b)
                res.append((pf, w, nsol))
            pf = float(np.median([r[0] for r in res])); w = float(np.median([r[1] for r in res])); ns = float(np.median([r[2] for r in res]))
            k_std = int(math.floor(math.pi / (4 * math.asin(math.sqrt(ns / 2 ** n)))))
            k_aq = int(math.floor(math.pi / (4 * math.asin(math.sqrt(min(1.0, pf)))))) if pf > 0 else None
            rows.append(dict(m=m, n=n, width=w, p_feas=pf, p_uniform=ns / 2 ** n, nsol=ns, k_std=k_std, k_aq=k_aq))
            print(rows[-1], flush=True)
    json.dump(rows, open('results/general_limits.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
