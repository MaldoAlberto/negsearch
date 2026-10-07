"""Exact optimum / worst feasible value of the ub=1 portfolio model by dynamic programming over periods.
Constraints are per period, couplings between periods are the transaction terms between x[t-1,d,i] and x[t,d,i]."""
import itertools, numpy as np


def period_states(P):
    """All feasible 0/1 vectors of one period (length 2n, local index d*n + i)."""
    n = P.na; out = []
    for L in range(n + 1):
        for S in range(n + 1):
            if not P.ok_counts(L, S): continue
            for a in itertools.combinations(range(n), L):
                for b in itertools.combinations(range(n), S):
                    v = np.zeros(2 * n, np.int8); v[list(a)] = 1; v[[n + j for j in b]] = 1; out.append(v)
    return np.array(out)


def optimum(P, sense='min'):
    X = period_states(P); n2 = 2 * P.na
    loc = {}; cross = {}
    for t in range(P.T):
        idx = [P.q(t, d, i) for d in (0, 1) for i in range(P.na)]
        h = P.h[idx].astype(float); Jm = np.zeros((n2, n2))
        pos = {g: k for k, g in enumerate(idx)}
        cr = np.zeros(n2)
        for (a, b), v in P.J.items():
            if a in pos and b in pos: Jm[pos[a], pos[b]] += v
            elif t > 0 and b in pos and a == P.q(t - 1, *divmod(pos[b], P.na)): cr[pos[b]] += v
        Xf = X.astype(float)
        loc[t] = Xf @ h + np.einsum('si,ij,sj->s', Xf, Jm, Xf); cross[t] = cr
    better = np.minimum if sense == 'min' else np.maximum
    pick = np.argmin if sense == 'min' else np.argmax
    V = loc[0].copy(); back = []
    Xf = X.astype(float)
    for t in range(1, P.T):
        M = V[:, None] + (Xf * cross[t][None, :]) @ Xf.T            # prev x cur
        bi = pick(M, axis=0); back.append(bi)
        V = M[bi, np.arange(len(X))] + loc[t]
    k = int(pick(V)); val = float(V[k]) + float(P.const)
    sol = [k]
    for bi in reversed(back): sol.append(int(bi[sol[-1]]))
    sol = sol[::-1]
    x = np.zeros(P.nq, np.int8)
    for t, s in enumerate(sol):
        for d in (0, 1):
            for i in range(P.na): x[P.q(t, d, i)] = X[s][d * P.na + i]
    return val, x
