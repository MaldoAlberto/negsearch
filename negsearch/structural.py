"""Constraint automata written from the problem STRUCTURE (no truth table), so the generic compiler scales.
build_structural(n, init, step, accept): forward-reachable states, backward liveness pruning, compact ids.
Also exact dynamic programmes over the automaton: number of feasible strings, marginals under A_q, optimum of a linear objective."""
from __future__ import annotations

import math

import numpy as np

from .automaton import Automaton


def build_structural(n, init, step, accept):
    """step(k, state, b) -> next state or None (dead); accept(state) -> bool at level n."""
    levels = [{init: 0}]
    nxt = []
    for k in range(n):
        cur = levels[-1]
        new, tr = {}, {}
        for s in cur:
            for b in (0, 1):
                t = step(k, s, b)
                if t is not None:
                    tr[(s, b)] = t
                    new.setdefault(t, len(new))
        levels.append(new); nxt.append(tr)
    # backward liveness
    alive = [None] * (n + 1)
    alive[n] = {s for s in levels[n] if accept(s)}
    for k in range(n - 1, -1, -1):
        alive[k] = {s for s in levels[k] if any(nxt[k].get((s, b)) in alive[k + 1] for b in (0, 1) if (s, b) in nxt[k])}
    # backward minimisation: states with the same (class of 0-child, class of 1-child) have the same feasible suffixes
    cls = [dict() for _ in range(n + 1)]
    for st in alive[n]:
        cls[n][st] = 0
    width = [0] * (n + 1); width[n] = 1 if alive[n] else 0
    sigs = [None] * (n + 1)
    for k in range(n - 1, -1, -1):
        sigmap = {}
        for st in alive[k]:
            sig = tuple(cls[k + 1][nxt[k][(st, b)]] if (st, b) in nxt[k] and nxt[k][(st, b)] in cls[k + 1] else -1 for b in (0, 1))
            cls[k][st] = sigmap.setdefault(sig, len(sigmap))
        width[k] = len(sigmap); sigs[k] = sigmap
    trans = []
    for k in range(n):
        T = [[-1, -1] for _ in range(width[k])]
        for sig, i in sigs[k].items():
            T[i] = list(sig)
        trans.append(T)
    A = Automaton(n=n, state_of=None, dead=None, live=None, cls=None, width=width, trans=trans)
    A.start = cls[0][init] if init in cls[0] else None
    assert A.start == 0 or A.start is None, 'start class must be id 0'
    return A


def count_feasible(A):
    """number of accepted strings (exact integer)"""
    cnt = [1] * A.width[A.n]
    for k in range(A.n - 1, -1, -1):
        cnt = [sum(cnt[t] for t in A.trans[k][s] if t >= 0) for s in range(A.width[k])]
    return cnt[0]


def marginals(A, q=0.5):
    """P(x_k = 1) for k = 0..n-1 under the walk of A_q, and the per-level state distribution."""
    dist = np.zeros(A.width[0]); dist[0] = 1.0
    p1 = []
    for k in range(A.n):
        nd = np.zeros(A.width[k + 1]); pk = 0.0
        for s in range(A.width[k]):
            t0, t1 = A.trans[k][s]
            if t0 >= 0 and t1 >= 0:
                nd[t0] += dist[s] * (1 - q); nd[t1] += dist[s] * q; pk += dist[s] * q
            elif t1 >= 0:
                nd[t1] += dist[s]; pk += dist[s]
            else:
                nd[t0] += dist[s]
        p1.append(pk); dist = nd
    return np.array(p1)


def linear_opt(A, c, sense='max'):
    """optimum of sum_k c_k x_k over accepted strings (exact DP over the automaton) and the argmax string"""
    n = A.n
    sgn = 1 if sense == 'max' else -1
    best = [[0.0] * A.width[n]]
    arg = [[()] * A.width[n]]
    for k in range(n - 1, -1, -1):
        bk, ak = [], []
        for s in range(A.width[k]):
            cand = []
            for b in (0, 1):
                t = A.trans[k][s][b]
                if t >= 0:
                    cand.append((sgn * c[k] * b + best[0][t], b, t))
            v, b, t = max(cand)
            bk.append(v); ak.append((b,) + arg[0][t])
        best.insert(0, bk); arg.insert(0, ak)
    return sgn * best[0][0], arg[0][0]


# ---- structural families ------------------------------------------------------------------------------------
def knapsack_automaton(w, W):
    n = len(w)
    return build_structural(n, 0, lambda k, s, b: (s + b * w[k]) if s + b * w[k] <= W else None, lambda s: True)


def cardinality_automaton(n, K):
    return build_structural(n, 0, lambda k, s, b: (s + b) if s + b <= K else None, lambda s: s == K)


def banded_is_automaton(n, d):
    """independent set on the band graph (i ~ j iff 0 < |i-j| <= d): state = last d bits"""
    def step(k, s, b):
        if b and any(s):
            return None
        return (s + (b,))[-d:] if len(s) >= d else s + (b,)
    return build_structural(n, (), lambda k, s, b: None if (b and any(s)) else ((s + (b,))[-d:]), lambda s: True)
