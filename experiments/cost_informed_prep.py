"""Can the cost function inform the preparation?  Ideal simulation on the 12 blocks of the 120-qubit instance (|F| <= 30 per block).
Start states with support exactly F:  uniform | product-tilt  w(x) = exp(-tau * sum_i g_i x_i), g = mean-field linear field of the block QUBO |  Boltzmann  w(x) = exp(-tau * c(x)).
tau from a small grid, chosen by the quality of the p=0 state (a classical computation on the block).  Then the usual p=1 layer (phase + pair-exchange mixers, light variant)
with gamma, beta trained on the state.  Writes results/cost_informed_prep.json."""
import os, sys, json, pickle, math
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); os.chdir(ROOT); sys.path.insert(0, ROOT)
import numpy as np
from scipy.optimize import minimize
import negsearch.hw100 as H
d = pickle.load(open('hardware_circuits/hw100/instance.pkl', 'rb')); inst = d['inst']
TAUS = [0, 0.25, 0.5, 1, 2, 4]
def evolve(b, psi0, par, variant='ours'):
    pairs = H.VARIANTS[variant]['pairs']; keep = H.VARIANTS[variant]['keep']
    idx = np.flatnonzero(b['mask']); ix = {tuple(int(v) for v in H.S_ALL[i]): k for k, i in enumerate(idx)}
    cph = H.qubo_energy(H.S_ALL[idx], b['hl'], H.pruned_Jl(b, keep))
    psi = psi0 * np.exp(-1j * par[0] * cph)
    for (j, k) in pairs:
        perm = []
        for i in idx:
            s_ = list(H.S_ALL[i]); s_[2 * j:2 * j + 2], s_[2 * k:2 * k + 2] = H.S_ALL[i][2 * k:2 * k + 2], H.S_ALL[i][2 * j:2 * j + 2]
            perm.append(ix[tuple(int(v) for v in s_)])
        inv = np.empty(len(perm), int); inv[np.array(perm)] = np.arange(len(perm))
        psi = math.cos(par[1]) * psi - 1j * math.sin(par[1]) * psi[inv]
    return np.abs(psi) ** 2
rows = []
for b in inst['blocks']:
    idx = np.flatnonzero(b['mask']); X = H.S_ALL[idx].astype(float); cv = b['cv'][idx]; span = b['cmax'] - b['cmin']
    q = lambda p: 1 - (float(p @ cv) - b['cmin']) / span
    mf = X.mean(0)                                                       # uniform marginals on F
    g = np.array(b['hl'], float).copy()
    for (i, j), w in H.pruned_Jl(b, 1.0).items(): g[i] += w * mf[j]; g[j] += w * mf[i]
    cn = (cv - b['cmin']) / span
    res = {'block': b['id'], 'F': int(len(idx))}
    for name, logw in (('uniform', lambda t: np.zeros(len(idx))), ('product', lambda t: -t * (X @ g) / max(np.abs(g).max(), 1e-9)), ('boltzmann', lambda t: -t * 4 * cn)):
        best = None
        for t in (TAUS if name != 'uniform' else [0]):
            w = np.exp(logw(t)); p0 = w / w.sum()
            if best is None or q(p0) > best[0]: best = (q(p0), t, p0)
        q0, t, p0 = best; psi0 = np.sqrt(p0).astype(complex)
        rng = np.random.default_rng(0); bb = None
        f = lambda par: float(evolve(b, psi0, par) @ cv)
        for r in range(8):
            o = minimize(f, rng.uniform(0, 1, 2), method='Nelder-Mead', options=dict(xatol=1e-3, fatol=1e-7, maxiter=120))
            if bb is None or o.fun < bb.fun: bb = o
        pf = evolve(b, psi0, bb.x)
        res[name] = dict(tau=t, q_p0=q0, q_p1=q(pf), popt_p0=float(p0[np.argmin(cv)]), popt_p1=float(pf[np.argmin(cv)]))
    rows.append(res)
S = {n: {k: float(np.mean([r[n][k] for r in rows])) for k in ('q_p0', 'q_p1', 'popt_p0', 'popt_p1')} for n in ('uniform', 'product', 'boltzmann')}
for n, v in S.items(): print(n, {k: round(x, 4) for k, x in v.items()})
json.dump(dict(rows=rows, mean=S), open('results/cost_informed_prep.json', 'w'))
