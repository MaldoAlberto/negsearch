"""Generality test 2 (QAOA): the SAME pipeline on the constraint families.
  penalty : |+>^n, diagonal phase of  c + lam * viol,  X mixer      (best lam of a grid, i.e. generous to the baseline)
  Aq-GM   : A_q|0>, diagonal phase of c only (no penalty), Grover mixer about A_q|0>   (generic: works for every F)
  Aq-NB   : A_q|0>, phase of c only, feasible-neighbourhood mixer (single flips + swaps inside F; numerics only)
Costs are rescaled so that the feasible range is [0,1].  Metrics from the exact final state.
Writes results/general_qaoa.json."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)
import json, math, time
import numpy as np
from scipy.optimize import minimize
from negsearch.automaton import (build_automaton, prep_state, gm_qaoa_state, x_qaoa_state, neighbour_hamiltonian,
                                 nb_qaoa_state)
from negsearch import families as fam
from experiments.general_grover import instances

LAMS = [0.5, 1.0, 2.0, 4.0]


def metrics(psi, P, c):
    pr = np.abs(psi) ** 2
    pr = pr / pr.sum()
    opt_set = P.good(0.0); top = P.good(0.1)
    q = float((pr * np.where(P.feas, 1 - c, 0.0)).sum())
    return dict(p_feas=float(pr[P.feas].sum()), p_opt=float(pr[opt_set].sum()), p_top=float(pr[top].sum()), quality=q)


def optimise(f, p, rng, restarts, prev=None, span=(2 * math.pi, 2 * math.pi)):
    best = None
    starts = []
    if prev is not None:                       # extend previous optimum by one layer
        g, b = prev[:p - 1], prev[p - 1:]
        starts.append(np.concatenate([g, [g[-1]], b, [b[-1]]]))
    while len(starts) < restarts:
        starts.append(np.concatenate([rng.uniform(0, span[0], p), rng.uniform(0, span[1], p)]))
    for x0 in starts:
        r = minimize(f, x0, method='BFGS', options=dict(maxiter=60, gtol=1e-5))
        if best is None or r.fun < best.fun:
            best = r
    return best


def run_instance(P, pmax, restarts, seed=0):
    rng = np.random.default_rng(seed)
    n = P.n
    A = build_automaton(P.feas, n)
    psi0 = prep_state(A, 0.5)
    span = P.worst - P.opt
    c = (P.cost - P.opt) / span                 # feasible range [0,1]
    out = dict(name=P.name, n=n, nF=P.nF, width=A.w, base={}, penalty={}, gm={}, nb={})
    out['base']['uniform'] = metrics(np.ones(2 ** n) / math.sqrt(2 ** n), P, c)
    out['base']['Aq'] = metrics(psi0, P, c)
    idx, H = neighbour_hamiltonian(P.feas, n)
    ev, evec = np.linalg.eigh(H)
    prevs = dict(gm=None, nb=None, **{f'pen{l}': None for l in LAMS})
    for p in range(1, pmax + 1):
        # --- penalty (best lam) ---
        best = None
        for lam in LAMS:
            cp = c + lam * P.viol
            f = lambda x: float((np.abs(x_qaoa_state(cp, x, n)) ** 2 * cp).sum())
            r = optimise(f, p, rng, restarts, prevs[f'pen{lam}'], span=(math.pi, math.pi))
            prevs[f'pen{lam}'] = r.x
            m = metrics(x_qaoa_state(cp, r.x, n), P, c); m['lam'] = lam
            if best is None or m['quality'] > best['quality']:
                best = m
        out['penalty'][p] = best
        # --- A_q + Grover mixer ---
        f = lambda x: float((np.abs(gm_qaoa_state(psi0, c, x)) ** 2 * c).sum())
        r = optimise(f, p, rng, restarts, prevs['gm'])
        prevs['gm'] = r.x
        out['gm'][p] = metrics(gm_qaoa_state(psi0, c, r.x), P, c)
        # --- A_q + neighbour mixer ---
        f = lambda x: float((np.abs(nb_qaoa_state(psi0, P.feas, c, x, ev, evec, idx, n)) ** 2 * c).sum())
        r = optimise(f, p, rng, restarts, prevs['nb'])
        prevs['nb'] = r.x
        out['nb'][p] = metrics(nb_qaoa_state(psi0, P.feas, c, r.x, ev, evec, idx, n), P, c)
    return out


def main(pmax=3, restarts=8):
    rows = []
    for P in instances():
        if P.n > 12:
            continue
        t = time.time()
        rows.append(run_instance(P, pmax, restarts))
        r = rows[-1]
        print(f"{P.name[:42]:42s} p=3: pen q={r['penalty'][pmax]['quality']:.3f} feas={r['penalty'][pmax]['p_feas']:.2f} | "
              f"GM q={r['gm'][pmax]['quality']:.3f} | NB q={r['nb'][pmax]['quality']:.3f} | A_q alone {r['base']['Aq']['quality']:.3f}  "
              f"({time.time() - t:.0f}s)", flush=True)
        json.dump(rows, open('results/general_qaoa.json', 'w'), indent=1)
    # n = 16 permutation instance, p <= 2
    P = fam.assignment_qap(4, 7)
    t = time.time()
    rows.append(run_instance(P, 2, 3))
    json.dump(rows, open('results/general_qaoa.json', 'w'), indent=1)
    print('QAP 4x4 done', time.time() - t)


if __name__ == '__main__':
    main()
