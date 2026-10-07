"""Quality of a QAOA layer whose objective phase is replaced by a single-qubit (mean-field linearised) layer: g_i = h_i + sum_j J_ij <x_j>_F, phase exp(-i gamma sum g_i x_i),
with the true cost used only to score the outcome.  Same feasible preparation and brick-wall pair-exchange mixers, simulated inside F.  This is the cheapest stand-in for a
Fourier-LCU-type decomposition of the objective (no two-qubit gate in the cost layer); it is NOT the LCU method itself.  Writes results/hybrid_quality_<tag>.json."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)
import json, math, os, warnings; warnings.filterwarnings('ignore')
import numpy as np
from scipy.optimize import minimize
import experiments.case_four as C
from negsearch.hw_large import swap_pairs
cp = C.core_problem(); bl, h1, J1, col, Fm = cp['bl'], cp['h1'], cp['J1'], cp['col'], cp['F']; f = len(cp['free']); nF = len(Fm)
cF = C.cost_of(Fm, h1, J1, col); cmin, cmax = cF.min(), cF.max(); cn = (cF - cmin) / (cmax - cmin); opt = np.abs(cF - cmin) < 1e-9
mean = Fm.mean(0); g = np.zeros(f)
for v, k in col.items(): g[k] += h1[v]
for (i, j), w in J1.items():
    if i in col and j in col: g[col[i]] += w * mean[col[j]]; g[col[j]] += w * mean[col[i]]
clin = Fm @ g
idx_of = {tuple(r): i for i, r in enumerate(Fm)}; PL = []
for blk in bl:
    if len(blk['F']) == 1: continue
    pr = swap_pairs(blk['size'], 'path'); pr = pr[::2] + pr[1::2]
    for (j, k) in pr:
        cols = [(col[blk['xq'][2 * j + d]], col[blk['xq'][2 * k + d]]) for d in (0, 1) if blk['xq'][2 * j + d] in col]; pm = np.empty(nF, int)
        for i, r in enumerate(Fm):
            r2 = r.copy()
            for a_, c_ in cols: r2[a_], r2[c_] = r[c_], r[a_]
            pm[i] = idx_of[tuple(r2)]
        PL.append(pm)
def run(phase_cost):
    def st(gm, bt):
        psi = np.ones(nF, complex) / math.sqrt(nF) * np.exp(-1j * gm * phase_cost)
        for pm in PL: psi = math.cos(bt) * psi - 1j * math.sin(bt) * psi[pm]
        return psi
    rng = np.random.default_rng(0); bo = None
    for _ in range(25):
        o = minimize(lambda x: float((np.abs(st(*x)) ** 2 * cF).sum()), [rng.uniform(-2, 2), rng.uniform(0, 1.6)], method='Nelder-Mead')
        if bo is None or o.fun < bo.fun: bo = o
    P = np.abs(st(*bo.x)) ** 2
    return dict(quality=float((P * (1 - cn)).sum()), popt=float(P[opt].sum()))
out = dict(f=f, nF=nF, uniform=dict(quality=float((1 - cn).mean()), popt=float(opt.mean())), full_cost=run(cF), single_qubit_cost=run(clin))
print(out, flush=True)
json.dump(out, open(f"results/hybrid_quality_{os.environ.get('CF_TAG', 'n64')}.json", 'w'), indent=1)
