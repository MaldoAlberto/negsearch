"""Quality comparison on the multi-constraint portfolio case (A=4 assets, T=2 periods, 2 sectors, n=16 variables, 521 feasible strings),
three instances.  Methods (all exact numerics, same objective):
  - 'A_q + Grover mixer'   : coherent automaton state, Grover mixer (feasible by construction)
  - 'conditioned + GM'     : interface (per-period position counts) drawn classically with weight |group|/|F|; within each group the
                             Grover mixer about the group's own state (conserves the interface, so feasible by construction)
  - 'penalty (best of 2)'  : cost + w * sum(violation^2), uniform start, X mixer, best of two weights
  - 'unbalanced'           : cost + sum_j (-l1 g_j + l2 g_j^2) over the inequality slacks g_j >= 0 (+ pair penalty), uniform start, X mixer
Quality = E[feasible * (1 - c)] with c the objective normalised on the feasible set; P(opt) = probability of an optimal string.
Angles: Nelder-Mead, 3 restarts, depth p = 1,2,3.  Writes results/large_case_quality.json."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)
import json, math, warnings; warnings.filterwarnings('ignore')
import numpy as np
from scipy.optimize import minimize
from negsearch.automaton import bits_of, gm_qaoa_state, x_qaoa_state

A, T, SECT, CAP, DMAX, BMAX = 4, 2, [2, 2], 1, 3, 3
Q = int(round(0.6 * T * BMAX)); n = 2 * A * T
X = bits_of(n)                                   # MSB-first, variable k = column k, order (t, i, d)
idx = lambda t, i, d: (t * A + i) * 2 + d
sector_of = np.repeat(np.arange(len(SECT)), SECT)


def constraint_terms():
    """list of slacks g >= 0 required (inequalities) and exclusivity violation count"""
    g = []; tot = 0
    for t in range(T):
        L = sum(X[:, idx(t, i, 0)] for i in range(A)); S = sum(X[:, idx(t, i, 1)] for i in range(A))
        tot = tot + L + S
        g += [L - S, DMAX - (L - S), BMAX - (L + S)]
        for s in range(len(SECT)):
            g.append(CAP - sum(X[:, idx(t, i, 0)] for i in range(A) if sector_of[i] == s))
    g.append(Q - tot)
    excl = sum(X[:, idx(t, i, 0)] * X[:, idx(t, i, 1)] for t in range(T) for i in range(A))
    return g, excl


G, EXCL = constraint_terms()
FEAS = (EXCL == 0) & np.all([g >= 0 for g in G], axis=0)
VIOL = sum(np.maximum(-g, 0) ** 2 for g in G) + EXCL
M = np.array([[sum(X[:, idx(t, i, d)] for i in range(A) for d in (0, 1)) for t in range(T)]]).squeeze(0).T if False else None
INTERFACE = np.stack([sum(X[:, idx(t, i, d)] for i in range(A) for d in (0, 1)) for t in range(T)], 1)


def make_cost(seed):
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.02, 0.12, A); Gm = rng.normal(size=(A, A)); Sg = Gm @ Gm.T / A * 0.05
    y = [X[:, [idx(t, i, 0) for i in range(A)]] - X[:, [idx(t, i, 1) for i in range(A)]] for t in range(T)]
    c = sum(np.einsum('ij,jk,ik->i', yt, Sg, yt) * 8 - yt @ mu * 10 for yt in y)
    c = c + 0.8 * sum(((y[t] - y[t - 1]) ** 2).sum(1) for t in range(1, T)) * 0.1
    return c


def run(seed):
    c = make_cost(seed)
    cmin, cmax = c[FEAS].min(), c[FEAS].max()
    cn = (c - cmin) / (cmax - cmin)                      # in [0,1] on the feasible set
    opt = FEAS & (np.abs(c - cmin) < 1e-12)
    uniform = np.ones(2 ** n) / math.sqrt(2 ** n)
    psiF = FEAS / math.sqrt(FEAS.sum())
    groups = {}
    for k in map(tuple, np.unique(INTERFACE[FEAS], axis=0)):
        mask = FEAS & np.all(INTERFACE == k, axis=1)
        groups[k] = (mask.sum() / FEAS.sum(), mask / math.sqrt(mask.sum()))

    def metrics(p):
        return dict(feas=float(p[FEAS].sum()), popt=float(p[opt].sum()), quality=float((p * FEAS * (1 - cn)).sum()))

    def best(fun, p):
        b = None
        rng = np.random.default_rng(seed + 11 * p)
        for r in range(3):
            x0 = rng.uniform(0, 2 * math.pi, 2 * p)
            res = minimize(lambda x: -fun(x)['quality'], x0, method='Nelder-Mead', options=dict(maxiter=150, xatol=1e-3, fatol=1e-5))
            if b is None or -res.fun > b[0]:
                b = (-res.fun, fun(res.x))
        return b[1]

    out = {}
    scale = np.ptp(c[FEAS])
    for p in (1, 2, 3):
        gm = lambda x: metrics(np.abs(gm_qaoa_state(psiF, cn, x)) ** 2)
        def cond(x):
            P = np.zeros(2 ** n)
            for w, ps in groups.values():
                P += w * np.abs(gm_qaoa_state(ps, cn, x)) ** 2
            return metrics(P)
        pen = None
        for w in (1, 4):
            cc = cn + w * VIOL
            r = best(lambda x, cc=cc: metrics(np.abs(x_qaoa_state(cc, x, n)) ** 2), p)
            if pen is None or r['quality'] > pen['quality']:
                pen = r
        l1, l2 = 0.96, 0.037
        unb = None
        for w in (1, 4):
            # unbalanced form on the slacks: -l1*g + l2*g^2 (g>=0 feasible side lowers the penalty); exclusivity quadratic
            ucost = cn + w * (sum(-l1 * g + l2 * g ** 2 for g in G) / 10 + EXCL)
            r = best(lambda x, uc=ucost: metrics(np.abs(x_qaoa_state(uc, x, n)) ** 2), p)
            if unb is None or r['quality'] > unb['quality']:
                unb = r
        out[p] = {'A_q + Grover mixer': best(gm, p), 'conditioned + GM': best(cond, p), 'penalty': pen, 'unbalanced': unb}
        print(f"seed {seed} p={p}: " + "  ".join(f"{k}: feas {v['feas']:.2f} Popt {v['popt']:.3f} q {v['quality']:.2f}" for k, v in out[p].items()), flush=True)
    out['A_q alone'] = metrics(np.abs(psiF) ** 2)
    out['uniform'] = metrics(np.abs(uniform) ** 2)
    return out


if __name__ == '__main__':
    print('n', n, '|F|', int(FEAS.sum()), 'uniform feasible share', FEAS.sum() / 2 ** n)
    res = {s: run(s) for s in (1, 2)}
    json.dump(res, open('results/large_case_quality.json', 'w'), indent=1, default=float)
