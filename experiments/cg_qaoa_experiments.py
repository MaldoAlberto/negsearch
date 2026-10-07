"""Simulation study for the tutorial: constraint-guided QAOA (CG-QAOA) vs penalty QAOA vs Hadfield QAOA
on QOBLIB maximum-independent-set instances, plus amplitude amplification vs classical sampling."""
import os as _os, sys as _sys
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..'))
_os.chdir(ROOT); _sys.path.insert(0, ROOT)
import numpy as np, json, time, math
from scipy.optimize import minimize
from negsearch.qaoa_mis import read_gph, MIS, prep_biased
Q = _os.environ.get('QOBLIB_MIS', 'qoblib/07-independentset/')   # path to a QOBLIB checkout (git clone https://github.com/ZIB-AOPT/QOBLIB qoblib)
sig = lambda z: 1 / (1 + np.exp(-z))

def opt(f, x0s, maxiter=250):
    best = None
    for x0 in x0s:
        r = minimize(f, x0, method='COBYLA', options=dict(maxiter=maxiter, rhobeg=0.4))
        if best is None or r.fun < best.fun: best = r
    return best

def best_known(name):
    for l in open(f'{Q}solutions/{name}.opt.sol'):
        if 'Objective value' in l: return int(l.split('=')[1])

import sys
SPEC = sys.argv[1] if len(sys.argv) > 1 else None; OUT = sys.argv[2] if len(sys.argv) > 2 else 'cg_results.json'
PMAX = int(sys.argv[3]) if len(sys.argv) > 3 else 3
if SPEC:
    insts = []
    for c in json.load(open(SPEC)):
        n0, E0 = read_gph(f"{Q}instances/{c['source']}.gph"); idx = {v - 1: i for i, v in enumerate(c['vertices'])}
        insts.append((c['name'], len(idx), [(idx[a], idx[b]) for a, b in E0 if a in idx and b in idx], c['opt']))
else:
    insts = []
    for name in ['farm', 'mammalia-kangaroo-interactions']:
        n, E = read_gph(f'{Q}instances/{name}.gph'); insts.append((name, n, E, best_known(name)))
out = {}
for name, n, E, bk in insts:
    g = MIS(n, E); assert g.opt == bk
    order = sorted(range(n), key=lambda v: (len(g.N[v]), v))          # low degree first (no knowledge of the solution)
    rng = np.random.default_rng(0)
    R = dict(n=n, m=len(E), opt=int(g.opt), n_feasible=int(g.feas.sum()),
             n_optimal=int((g.feas & (g.size == g.opt)).sum()), order=order)
    R['uniform'] = dict(p_feas=float(g.feas.mean()), p_opt=R['n_optimal'] / 2 ** n, ratio=float((g.size * g.feas).mean() / g.opt))
    R['prep_q0.5'] = g.metrics(prep_biased(g, order, 0.5))
    print(name, R['uniform'], R['prep_q0.5'], flush=True)
    prev = {}
    for p in range(1, PMAX + 1):
        t0 = time.time(); res = {}
        def warm(key, dim, extra=0):
            starts = [rng.uniform(0, np.pi, dim + extra) for _ in range(3)]
            if key in prev:                       # parameter transfer from depth p-1 (append a layer)
                x = prev[key]; pp = p - 1
                g_, b_ = list(x[:pp]), list(x[pp:2 * pp]); tail = list(x[2 * pp:])
                starts.append(np.array(g_ + [g_[-1]] + b_ + [b_[-1]] + tail))
            return starts
        # penalty QAOA (QUBO, uniform start, X mixer), lambda in {2,3}
        best = None
        for lam in (2.0, 3.0):
            C = -g.size + lam * g.viol
            r = opt(lambda x: float((np.abs(g.penalty_state(x, lam)) ** 2 * C).sum()), warm(f'pen{lam}', 2 * p))
            prev[f'pen{lam}'] = r.x; m = g.metrics(g.penalty_state(r.x, lam))
            if best is None or m['ratio'] > best['ratio']: best = dict(lam=lam, **m)
        res['penalty'] = best
        # Hadfield constraint-preserving QAOA (|0>, neighbour-controlled RX)
        r = opt(lambda x: -float((np.abs(g.hadfield_state(x)) ** 2 * g.size).sum()), warm('had', 2 * p))
        prev['had'] = r.x; res['hadfield'] = g.metrics(g.hadfield_state(r.x))
        # CG-QAOA with fixed q = 0.5
        psi05 = prep_biased(g, order, 0.5)
        r = opt(lambda x: -float((np.abs(g.gm_state(x, psi05)) ** 2 * g.size).sum()), warm('cg05', 2 * p))
        prev['cg05'] = r.x; res['cg_q0.5'] = g.metrics(g.gm_state(r.x, psi05))
        # CG-QAOA with variational coin bias q = sigmoid(z)
        def fq(x):
            psi0 = prep_biased(g, order, sig(x[-1])); return -float((np.abs(g.gm_state(x[:-1], psi0)) ** 2 * g.size).sum())
        r = opt(fq, warm('cgq', 2 * p, extra=1))
        prev['cgq'] = r.x; q = float(sig(r.x[-1]))
        res['cg_qvar'] = dict(q=q, **g.metrics(g.gm_state(r.x[:-1], prep_biased(g, order, q))))
        res['prep_qvar_alone'] = dict(q=q, **g.metrics(prep_biased(g, order, q)))
        res['params'] = {k: list(map(float, v)) for k, v in prev.items()}
        R[f'p{p}'] = res
        print(f' p={p} ({time.time()-t0:.0f}s)', {k: (v if k == 'params' else {kk: round(vv, 4) for kk, vv in v.items()}) for k, v in res.items() if k != 'params'}, flush=True)
    # amplitude amplification over A_q with threshold oracle |x| >= best known  vs classical sampling of A_q
    amp = []
    for q in (0.5, 0.6, 0.7, 0.8, 0.9):
        psi0 = prep_biased(g, order, q); good = g.feas & (g.size >= g.opt)
        P = float((np.abs(psi0[good]) ** 2).sum()); th = math.asin(math.sqrt(P)); k = int(math.floor(math.pi / (4 * th)))
        psi = psi0.copy()                                   # verify numerically: k rounds of S_good then reflection about psi0
        for _ in range(k):
            psi = np.where(good, -psi, psi); psi = 2 * psi0 * np.vdot(psi0, psi) - psi
        succ = float((np.abs(psi[good]) ** 2).sum())
        amp.append(dict(q=q, P_prep=P, classical_samples=1 / P, k=k, success_sim=succ,
                        success_theory=math.sin((2 * k + 1) * th) ** 2, quantum_prep_calls=(2 * k + 1) / succ))
        print('  amp', amp[-1], flush=True)
    R['amplification'] = amp
    out[name] = R
    json.dump(out, open(OUT, 'w'), indent=1)
