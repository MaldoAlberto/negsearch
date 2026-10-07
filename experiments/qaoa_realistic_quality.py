"""Exact quality of the optimised QAOA variants (conditioned multi-constraint case), averaged over ALL interface tuples with their weights.
Variants: block Grover mixer (GM), pair-exchange mixer on all pairs / on a path, with the full objective or with small couplings pruned
(the circuit uses the pruned phase, the quality is always measured with the TRUE objective).  Quality = E[1 - (c-opt)/(worst-opt)] (all samples are
feasible by construction); P(opt) = probability of an optimal string.  Angles: Nelder-Mead, 4 restarts.  Writes results/qaoa_realistic_quality.json."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)
import json, warnings; warnings.filterwarnings('ignore')
import numpy as np
from scipy.optimize import minimize
import experiments.hardware_large_case as H
from negsearch.hw_large import interface_tuples, qubo_cost, tensor_qaoa_swap, prune


def run(A, T, S, depths=(1, 2), topk=None, restarts=4):
    H.A, H.T, H.SECT, H.NV = A, T, S, 2 * A * T
    h, J = qubo_cost(A, T, H.SEED)
    tuples, _ = interface_tuples(A, T, S)
    if topk:
        tuples = sorted(tuples, key=lambda t: -t[1])[:topk]; tw = sum(w for _, w in tuples); tuples = [(i, w / tw) for i, w in tuples]
    data = []
    for iface, w in tuples:
        bl = H.tuple_support(iface); X, combos = H.full_strings(bl); data.append((w, bl, X, combos, H.cost_of(X, h, J)))
    opt = min(d[4].min() for d in data); worst = max(d[4].max() for d in data)

    def phase_cost(X, frac):
        if frac == 0:
            return None
        hp, Jp = h.copy(), dict(J)
        m = max(abs(v) for v in Jp.values())
        Jp = {k: v for k, v in Jp.items() if abs(v) >= frac * 4 * m} if False else Jp   # pruning is done on the Ising coefficients below
        return None

    from negsearch.hw_large import ising_from_qubo
    co = ising_from_qubo(h, J)
    results = {}
    for name, mixer, frac in [('GM', 'gm', 0), ('SWAP all', 'all', 0), ('SWAP path', 'path', 0), ('SWAP path + prune 25%', 'path', 0.25)]:
        cop = prune(co, frac) if frac else co
        # cost used in the phase layer: Ising polynomial evaluated on the strings
        def ph(X):
            z = 1 - 2 * X; c = np.zeros(len(X))
            for k, v in cop.items():
                c += v * np.prod(z[:, list(k)], axis=1)
            return c
        pc = [ph(d[2]) for d in data]

        def metrics(params):
            q = 0; po = 0
            for (w, bl, X, combos, c), phc in zip(data, pc):
                if mixer == 'gm':
                    P = H.tensor_qaoa(bl, combos, phc, params)[tuple(combos.T)]
                else:
                    P = tensor_qaoa_swap(bl, combos, phc, params, mixer)[tuple(combos.T)]
                cn = (c - opt) / (worst - opt)
                q += w * (P * (1 - cn)).sum(); po += w * P[np.abs(c - opt) < 1e-9].sum()
            return q, po
        for p in depths:
            rng = np.random.default_rng(3 + p); best = None
            for r in range(restarts):
                x0 = np.concatenate([rng.uniform(-1.5, 1.5, p), rng.uniform(0, 1.5, p)])
                res = minimize(lambda x: -metrics(x)[0], x0, method='Nelder-Mead', options=dict(maxiter=70, xatol=1e-3, fatol=1e-5))
                if best is None or res.fun < best.fun:
                    best = res
            q, po = metrics(best.x)
            results[f'{name} p={p}'] = dict(quality=q, popt=po)
            print(f"A={A},T={T} {name:24s} p={p}: quality {q:.3f}  P(opt) {po:.4f}", flush=True)
    q0 = sum(d[0] * (1 - (d[4] - opt) / (worst - opt)).mean() for d in data)
    results['preparation only'] = dict(quality=q0)
    print(f"A={A},T={T} preparation only: quality (uniform within tuples) {q0:.3f}")
    return results


if __name__ == '__main__':
    import sys as _s
    if len(_s.argv) > 1 and _s.argv[1] == 'n16':          # the 8 most probable interface tuples (renormalised), p=1, 3 restarts
        out = {'n16_top8': run(4, 2, [2, 2], depths=(1,), topk=8, restarts=3)}
        json.dump(out, open('results/qaoa_realistic_quality_n16.json', 'w'), indent=1)
    else:
        out = {'n8': run(4, 1, [2, 2]), 'n16': run(4, 2, [2, 2])}
        json.dump(out, open('results/qaoa_realistic_quality.json', 'w'), indent=1)
