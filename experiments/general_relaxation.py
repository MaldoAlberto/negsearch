"""Generality test 4: width-vs-completeness dial.  Exact automaton (width w) gives support = F.  A cheaper, sound but
incomplete automaton (a lower bound on the knapsack weight, kept in units of r) forces only when the violation is certain,
so its support is a superset of F (dead ends).  We report width, P(feasible) of A_q, and the amplification iterations
needed for the optimal solutions.  Writes results/general_relaxation.json."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)
import json, math
import numpy as np
from negsearch.automaton import bits_of, best_k
from negsearch import families as fam


def relaxed_mu(w, W, r, q=0.5):
    """Walk with abstract state a_k = sum floor(w_i/r) over chosen items (lower bound of weight/r).  x_k=1 is excluded only if
    r*(a_k+floor(w_k/r)) > W (certainly infeasible).  Returns mu over all 2^n strings and the number of abstract states."""
    n = len(w)
    X = bits_of(n)
    mu = np.ones(2 ** n)
    a = np.zeros(2 ** n, int)
    widths = []
    for k in range(n):
        wk = w[k] // r
        can1 = r * (a + wk) <= W
        # abstract live set of bit 0 is always alive (add nothing); bit 1 alive iff can1
        b = X[:, k]
        both = can1
        fac = np.where(both, np.where(b == 1, q, 1 - q), 1.0)
        ok = np.where(b == 1, can1, True)
        mu *= np.where(ok, fac, 0.0)
        # reachable abstract states among strings with positive weight so far
        widths.append(len(np.unique(a[mu > 0])))
        a = a + b * wk
    return mu, max(widths)


def main():
    out = []
    for seed in range(6):
        P = fam.knapsack(12, 100 + seed, wmax=12, frac=0.45)
        w = P.meta['w']; W = P.meta['W']
        G = P.good(0.0)
        for r in (1, 2, 3, 4, 6, 12):
            mu, wd = relaxed_mu(np.array(w), W, r)
            pf = float(mu[P.feas].sum()); a = float(mu[G].sum())
            out.append(dict(seed=seed, n=P.n, W=W, r=r, width=wd, p_feas=pf, a_opt=a, k_amp=best_k(a),
                            k_std=best_k(G.sum() / 2 ** P.n), sum_mu=float(mu.sum()), support_ge_F=bool((mu[P.feas] > 0).all())))
    json.dump(out, open('results/general_relaxation.json', 'w'), indent=1)
    print('r  width  P(feas of A_q)  a(opt)   k_amp   k_std   (median over 6 instances, n=12)')
    for r in (1, 2, 3, 4, 6, 12):
        R = [o for o in out if o['r'] == r]
        m = lambda k: float(np.median([o[k] for o in R]))
        print(f"{r:2d} {m('width'):5.0f} {m('p_feas'):12.3f} {m('a_opt'):12.2e} {m('k_amp'):6.0f} {m('k_std'):7.0f}  sound={all(o['support_ge_F'] for o in R)}")


if __name__ == '__main__':
    main()
