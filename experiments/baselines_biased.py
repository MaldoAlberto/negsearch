"""Fairness check for the cost-informed preparation: give the warm-started baselines the SAME classical information.
Their product start RY(2 asin sqrt(p_i)) uses p_i = marginals of the product-tilted distribution on F (tau=4, mean-field field g of the block), instead of k/m.
Ideal simulation, 12 blocks of the 120-qubit instance; trained angles (restarts as in negsearch.hw100).  Reports ideal feasible fraction and quality among feasible samples.
Writes results/baselines_biased.json."""
import os, sys, json, pickle
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); os.chdir(ROOT); sys.path.insert(0, ROOT)
import numpy as np
import negsearch.hw100 as H
d = pickle.load(open('hardware_circuits/hw100/instance.pkl', 'rb')); inst = d['inst']
orig = H.warm_p; cache = {}
def tilted_marginals(b, tau=4.0):
    idx = np.flatnonzero(b['mask']); X = H.S_ALL[idx].astype(float)
    mf = X.mean(0); g = np.array(b['hl'], float).copy()
    for (i, j), w in H.pruned_Jl(b, 1.0).items(): g[i] += w * mf[j]; g[j] += w * mf[i]
    w = np.exp(-tau * (X @ g) / max(np.abs(g).max(), 1e-9)); w /= w.sum()
    return np.clip(w @ X, 1e-3, 1 - 1e-3).tolist()
res = {m: {'plain': [], 'biased': []} for m in ('lcu', 'unbalanced', 'xy')}
for b in inst['blocks']:
    for mode in ('plain', 'biased'):
        H.warm_p = orig if mode == 'plain' else (lambda bb, _p=tilted_marginals(b): _p)
        for m in res:
            par = H.train_baseline(b, m, seed=0); P = H.baseline_probs(b, m, par); mt = H.metrics(b, P)
            res[m][mode].append((mt['feas'], mt['quality']))
    print(b['id'], flush=True)
H.warm_p = orig
out = {m: {mode: dict(feas=float(np.mean([x[0] for x in v])), quality=float(np.mean([x[1] for x in v]))) for mode, v in r.items()} for m, r in res.items()}
print(json.dumps(out, indent=1)); json.dump(out, open('results/baselines_biased.json', 'w'))
