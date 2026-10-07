"""Retrain gamma, beta of our p=1 layer ON THE ACTUAL CIRCUIT (exact state-vector of the compiled-logical block circuit) for both preparations:
exact controlled swaps and relative-phase Toffoli swaps (negsearch.dicke_opt.REL).  All 12 blocks of the 120-qubit instance, ours (light) and ours_full.
Quality = 1 - (mean cost among feasible samples - cmin)/(cmax - cmin) of the ideal output distribution (feasible mass is 1 in both).  Writes results/retrain_relphase.json."""
import os, sys, json, pickle
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); os.chdir(ROOT); sys.path.insert(0, ROOT)
import numpy as np
from scipy.optimize import minimize
import negsearch.hw100 as H
from negsearch import dicke_opt as D
d = pickle.load(open('hardware_circuits/hw100/instance.pkl', 'rb')); inst, params = d['inst'], d['params']
RES = int(os.environ.get('RT_RESTARTS', 5)); out = {}
for rel in (False, True):
    D.REL['on'] = rel; rng = np.random.default_rng(0); rows = {'ours': [], 'ours_full': []}
    for b in inst['blocks']:
        idx = np.flatnonzero(b['mask']); cv = b['cv']; span = b['cmax'] - b['cmin']
        for method in rows:
            def f(p):
                par = dict(params[b['id']]); par[method] = list(p)
                pr = H.probs_from_circuit(H.block_circuit(b, method, par)); return float(pr[idx] @ cv[idx] / pr[idx].sum())
            best = None
            for r in range(RES):
                o = minimize(f, rng.uniform(0, 1, 2), method='Nelder-Mead', options=dict(xatol=1e-3, fatol=1e-7, maxiter=100))
                if best is None or o.fun < best.fun: best = o
            rows[method].append(1 - (best.fun - b['cmin']) / span)
    out['rel' if rel else 'exact'] = {k: dict(mean=float(np.mean(v)), per_block=[float(x) for x in v]) for k, v in rows.items()}
    print('rel' if rel else 'exact', {k: round(out['rel' if rel else 'exact'][k]['mean'], 3) for k in rows}, flush=True)
json.dump(out, open('results/retrain_relphase.json', 'w'))
