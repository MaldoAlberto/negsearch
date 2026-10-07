"""Small constraint-optimisation families used to test generality (all built as explicit Problem objects).
Every family has a different constraint type; the SAME pipeline (automaton -> A_q -> Grover / QAOA) is run on all."""
from __future__ import annotations

import itertools

import numpy as np

from .automaton import Problem, bits_of


def _mk(name, n, feas, cost, viol, **meta):
    return Problem(name, n, feas.astype(bool), cost.astype(float), viol.astype(float), meta)


def cardinality_qp(n=10, k=4, seed=0):
    """Choose exactly k of n assets, minimise risk x^T S x - mu^T x   (count constraint, quadratic objective)."""
    rng = np.random.default_rng(seed)
    X = bits_of(n)
    mu = rng.uniform(0.02, 0.12, n)
    G = rng.normal(size=(n, n)); S = G @ G.T / n * 0.02
    cost = np.einsum('ij,jk,ik->i', X, S, X) * 10 - X @ mu * 10
    cnt = X.sum(1)
    return _mk(f'cardinality ({k} of {n})', n, cnt == k, cost, (cnt - k) ** 2, kind='count')


def knapsack(n=10, seed=1, wmax=9, frac=0.45):
    """0/1 knapsack with DIFFERENT weights: max value s.t. sum w_i x_i <= W   (inequality, state = partial weight)."""
    rng = np.random.default_rng(seed)
    X = bits_of(n)
    w = rng.integers(1, wmax + 1, n); v = rng.integers(2, 20, n)
    W = int(frac * w.sum())
    wt = X @ w
    return _mk(f'knapsack (weights {w.tolist()}, W={W})', n, wt <= W, -(X @ v).astype(float),
               np.maximum(wt - W, 0) ** 2, kind='knapsack', w=w.tolist(), v=v.tolist(), W=W)


def mis(n=10, p=0.35, seed=2):
    """Maximum weighted independent set on G(n,p)  (pairwise conflicts)."""
    rng = np.random.default_rng(seed)
    E = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p]
    X = bits_of(n)
    wts = rng.integers(1, 6, n)
    viol = sum((X[:, i] & X[:, j]) for i, j in E)
    return _mk(f'weighted MIS (G({n},{p}), {len(E)} edges)', n, viol == 0, -(X @ wts).astype(float), viol,
               kind='graph', edges=E)


def set_packing(n=10, m=7, seed=3):
    """Max-weight set packing: n sets over m elements, each element covered at most once (overlap conflicts)."""
    rng = np.random.default_rng(seed)
    sets = [rng.choice(m, size=rng.integers(2, 4), replace=False) for _ in range(n)]
    X = bits_of(n)
    wts = rng.integers(2, 9, n)
    cover = np.zeros((2 ** n, m), int)
    for i, s in enumerate(sets):
        for e in s:
            cover[:, e] += X[:, i]
    viol = np.maximum(cover - 1, 0).sum(1)
    return _mk(f'set packing ({n} sets, {m} elements)', n, viol == 0, -(X @ wts).astype(float), viol,
               kind='packing')


def exact_cover(n=10, m=6, seed=4, tries=200):
    """Weighted exact cover (every element covered exactly once): feasible set can be tiny."""
    rng = np.random.default_rng(seed)
    for _ in range(tries):
        sets = [rng.choice(m, size=rng.integers(1, 4), replace=False) for _ in range(n)]
        X = bits_of(n)
        cover = np.zeros((2 ** n, m), int)
        for i, s in enumerate(sets):
            for e in s:
                cover[:, e] += X[:, i]
        viol = np.abs(cover - 1).sum(1)
        if 3 <= (viol == 0).sum() <= 40:
            wts = rng.integers(1, 9, n)
            return _mk(f'exact cover ({n} sets, {m} elements)', n, viol == 0, (X @ wts).astype(float), viol,
                       kind='cover')
    raise RuntimeError


def colouring(nv=5, seed=5, p=0.55):
    """Proper 3-colouring of a graph (2 bits per vertex, code 11 forbidden), min total colour cost."""
    rng = np.random.default_rng(seed)
    E = [(i, j) for i in range(nv) for j in range(i + 1, nv) if rng.random() < p]
    n = 2 * nv
    X = bits_of(n)
    col = X[:, 0::2] * 2 + X[:, 1::2]                 # value 0..3 per vertex, 3 invalid
    viol = (col == 3).sum(1) + sum((col[:, i] == col[:, j]) & (col[:, i] != 3) for i, j in E)
    cc = rng.integers(1, 6, (nv, 3))
    cost = np.zeros(2 ** n)
    for v in range(nv):
        cost += np.where(col[:, v] < 3, cc[v][np.minimum(col[:, v], 2)], 0)
    return _mk(f'3-colouring ({nv} vertices, {len(E)} edges)', n, viol == 0, cost, viol, kind='colouring')


def assignment_qap(m=3, seed=6):
    """Quadratic assignment, m x m permutation matrices (row-major bits)."""
    rng = np.random.default_rng(seed)
    n = m * m
    X = bits_of(n)
    R = X.reshape(-1, m, m)
    viol = (np.abs(R.sum(2) - 1)).sum(1) + (np.abs(R.sum(1) - 1)).sum(1)
    Fm = rng.integers(0, 5, (m, m)); Fm = np.triu(Fm, 1); Fm = Fm + Fm.T
    Dm = rng.integers(1, 6, (m, m)); Dm = np.triu(Dm, 1); Dm = Dm + Dm.T
    cost = np.zeros(2 ** n)
    for i, j in itertools.product(range(m), repeat=2):
        for a, b in itertools.product(range(m), repeat=2):
            cost += Fm[i, j] * Dm[a, b] * R[:, i, a] * R[:, j, b]
    return _mk(f'quadratic assignment ({m}x{m} permutations)', n, viol == 0, cost, viol, kind='permutation')


def all_families():
    return [cardinality_qp(10, 4, 0), knapsack(10, 1), mis(10, 0.35, 2), set_packing(10, 7, 3), exact_cover(10, 6, 4),
            colouring(5, 5), assignment_qap(3, 6)]


def hardware_instance():
    """The 6-variable exact-cover instance of the hardware example: 3 feasible covers, one optimum, a_q(optimum) = 1/4,
    A_q needs 8 qubits (6 decisions + 2 automaton-register qubits)."""
    from .automaton import build_automaton, compile_Aq, mu_q
    for s in range(60):
        P = exact_cover(6, 4, 100 + s)
        if P.nF < 3:
            continue
        A = build_automaton(P.feas, P.n)
        mu, _ = mu_q(A, 0.5)
        G = P.good(0.0)
        if abs(mu[G].sum() - 0.25) < 1e-9 and compile_Aq(A, 0.5)[0].num_qubits == 8:
            return P
    raise RuntimeError('instance not found')
