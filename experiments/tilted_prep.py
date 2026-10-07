"""Cost-informed preparation with the SAME gates: only the rotation angles of the Dicke blocks change (support stays F, amplitudes become positive and biased).
Per block of the 120-qubit instance: target amplitudes sqrt(w(x)) on F, with w = product tilt exp(-tau sum_i g_i x_i) (g: mean-field linear field of the block QUBO) or Boltzmann exp(-tau c(x)),
tau chosen on a grid by the quality of the target; the angles of the symmetric block preparation (Dicke on longs, Dicke on compressed shorts, controlled-swap decompression)
are fitted by least squares to the target; then the p=1 layer (phase + pair-exchange mixers) is trained on the actual circuit state vector.
Everything is exact simulation of the logical circuit; the CX count is unchanged by construction (reported).  Writes results/tilted_prep.json."""
import os, sys, json, pickle, math
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); os.chdir(ROOT); sys.path.insert(0, ROOT)
import numpy as np
from scipy.optimize import least_squares, minimize
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
import negsearch.hw100 as H
from negsearch import dicke_opt as D
d = pickle.load(open('hardware_circuits/hw100/instance.pkl', 'rb')); inst, params = d['inst'], d['params']
D.REL['on'] = False; D.GIV['on'] = False
orig_cry, orig_ccry = D.cry, D.ccry
def prep_with(m, l, s, angles=None):
    """symmetric block prep (4-CX exact ccRY) with the Dicke rotation angles taken from `angles` (None: record the default ones)"""
    rec = []; it = iter(angles) if angles is not None else None
    def cry(qc, th, c, t):
        a = next(it) if it is not None else th; rec.append(th); orig_cry(qc, a, c, t)
    def ccry(qc, th, a_, b_, t):
        a = next(it) if it is not None else th; rec.append(th); orig_ccry(qc, a, a_, b_, t)
    D.cry, D.ccry, D.MODE['ccry'] = cry, ccry, ccry
    try: qc = D.symmetric_block_prep_opt(m, l, s, cheap=False)
    finally: D.cry, D.ccry = orig_cry, orig_ccry
    D.MODE['ccry'] = orig_ccry
    return qc, np.array(rec)
LOGB = ['cx', 'rz', 'sx', 'x']
def cx(qc): return int(transpile(qc, basis_gates=LOGB, optimization_level=3, seed_transpiler=1).count_ops().get('cx', 0))
TAUS = [0.25, 0.5, 1, 2, 4]
rows = []
for b in inst['blocks']:
    l, s = b['ls']; M = H.M
    idx = np.flatnonzero(b['mask']); X = H.S_ALL[idx].astype(float); cv = b['cv'][idx]; span = b['cmax'] - b['cmin']
    quality = lambda p: 1 - (float(p @ cv) - b['cmin']) / span
    mf = X.mean(0); g = np.array(b['hl'], float).copy()
    for (i, j), w in H.pruned_Jl(b, 1.0).items(): g[i] += w * mf[j]; g[j] += w * mf[i]
    cn = (cv - b['cmin']) / span
    qc0, th0 = prep_with(M, l, s)
    pidx = (H.S_ALL[idx].astype(int) * (1 << np.arange(H.NB_Q))).sum(1)      # little-endian index of every feasible string
    res = dict(block=b['id'], F=int(len(idx)), cx_prep=cx(qc0), n_angles=int(len(th0)))
    for name, logw in (('product', lambda t: -t * (X @ g) / max(np.abs(g).max(), 1e-9)), ('boltzmann', lambda t: -t * 4 * cn)):
        bt = max(TAUS, key=lambda t: quality(np.exp(logw(t)) / np.exp(logw(t)).sum()))
        w = np.exp(logw(bt)); ptar = w / w.sum(); amp = np.sqrt(ptar)
        def resid(a):
            sv = Statevector(prep_with(M, l, s, a)[0]).data
            return sv[pidx].real - amp if False else np.abs(sv[pidx]) - amp
        rng = np.random.default_rng(0); best = None
        for r in range(6):
            o = least_squares(resid, th0 + (0 if r == 0 else rng.normal(0, 0.3, len(th0))), max_nfev=120)
            if best is None or o.cost < best.cost: best = o
            if best.cost < 1e-10: break
        qc1, _ = prep_with(M, l, s, best.x); sv = Statevector(qc1).data; preal = np.abs(sv[pidx]) ** 2
        mass = float(preal.sum()); preal = preal / mass
        # train gamma, beta of the p=1 layer on the circuit
        H_sym = H.symmetric_block_prep
        H.symmetric_block_prep = lambda m_, l_, s_, _q=qc1: _q
        try:
            f = lambda p: (lambda pr: float(pr[b['mask']] @ b['cv'][b['mask']] / pr[b['mask']].sum()))(H.probs_from_circuit(H.block_circuit(b, 'ours', {b['id']: {}, 'ours': list(p)} if False else {'ours': list(p), 'ours_full': [0, 0], 'ours_p0': [0, 0]})))
            bb = None
            for r in range(5):
                o = minimize(f, rng.uniform(0, 1, 2), method='Nelder-Mead', options=dict(xatol=1e-3, fatol=1e-7, maxiter=100))
                if bb is None or o.fun < bb.fun: bb = o
        finally: H.symmetric_block_prep = H_sym
        res[name] = dict(tau=bt, fit_cost=float(best.cost), q_target=quality(ptar), q_p0=quality(preal), min_prob=float(preal.min()), mass=mass, q_p1=1 - (bb.fun - b['cmin']) / span, popt_p0=float(preal[np.argmin(cv)]))
    rows.append(res); print(res['block'], {k: (round(v['q_p0'], 3), round(v['q_p1'], 3), round(v['min_prob'], 5)) for k, v in res.items() if isinstance(v, dict)}, flush=True)
S = {n: {k: float(np.mean([r[n][k] for r in rows])) for k in ('q_target', 'q_p0', 'q_p1', 'popt_p0', 'min_prob', 'fit_cost')} for n in ('product', 'boltzmann')}
print(S); json.dump(dict(rows=rows, mean=S), open('results/tilted_prep.json', 'w'))
