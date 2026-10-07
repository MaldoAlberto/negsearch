"""Scaling on QOBLIB portfolio instances with Qiskit Aer's matrix-product-state simulator (12 -> 300 decision qubits).
Methods (all p = 1).  Angles: grid around the optimum transferred from the exactly optimised small instances (phase in
units of 1/max|h|), chosen on 300-shot MPS estimates of the method's own cost; 2000 fresh shots for the metrics:
  uniform             classical random bitstrings
  A_q classical       counts (L,S) drawn from the negation-forced weights + uniform subset (classical sampling)
  unbalanced p=1      penalty QAOA without slack (|+>, X mixer), weight alpha = 1/16 (best on the small instances)
  hybrid p=1          coherent 'sector + Dicke' preparation (no ancillas) + linear phase + XY on a path
Exact optimum / worst feasible value by dynamic programming (negsearch/portfolio_dp.py).
CZ / depth on ibm_fez (FakeFez) when the circuit fits on 156 qubits.
Usage: python experiments/portfolio_scaling_mps.py [keys...]"""
import os as _os, sys as _sys
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..'))
_os.chdir(ROOT); _sys.path.insert(0, ROOT)
import json, time, itertools, numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeFez
from negsearch.portfolio import Instance, Portfolio
from negsearch.portfolio_dp import optimum
from negsearch.portfolio_circuits import qaoa_circuit
from negsearch.portfolio_sector import sector_weights, sector_hybrid
from experiments.portfolio_midmeasure import sector_circuit

DATA = 'instances/qoblib_portfolio'
INST = {  # key: (dir, periods, B, lambda, cs2)
    'a003_t02': ('po_a003_t02_orig', None, 3, '1e-05', 2),
    'a004_t04': ('po_a004_t04_orig', None, 4, '2e-05', 3),
    'a005_t04': ('po_a005_t04_orig', None, 4, '4e-05', 3),
    'a010_t04': ('po_a010_t10_orig', 4, 4, '1e-05', 3),
    'a010_t07': ('po_a010_t10_orig', 7, 4, '1e-05', 3),
    'a010_t10': ('po_a010_t10_orig', None, 4, '1e-05', 3),
    'a010_t15': ('po_a010_t15_orig', None, 4, '1e-05', 3),
}
BASIS = ['cx', 'rz', 'ry', 'rx', 'h', 'x', 'sx', 'rzz']
MPS = AerSimulator(method='matrix_product_state', matrix_product_state_max_bond_dimension=64,
                   matrix_product_state_truncation_threshold=1e-8)
BK = FakeFez()


def load_big(key):
    d, T, B, lam, cs2 = INST[key]
    P = Portfolio(Instance(f'{DATA}/{d}', periods=T), B, lam, cs2=cs2, order='shorts_first'); P.xy_ring = False
    return P


def sample(qc, shots, seed):
    qc = qc.copy(); qc.measure_all()
    c = MPS.run(transpile(qc, basis_gates=BASIS, optimization_level=1), shots=shots, seed_simulator=seed).result().get_counts()
    bits, w = [], []
    for k, v in c.items():
        bits.append([int(ch) for ch in k.replace(' ', '')[::-1]]); w.append(v)
    return np.array(bits, np.int8), np.array(w)


def metrics(P, bits, w, fopt, fworst):
    x = bits[:, :P.nq]; ok = P.feasible(x); f = P.energy(x).astype(float); tot = w.sum()
    g = np.where(ok, (fworst - f) / (fworst - fopt), 0.0)
    best = f[ok].min() if ok.any() else np.nan
    return dict(p_feas=float(w[ok].sum() / tot), quality=float((w * g).sum() / tot),
                best_gap=float((best - fopt) / (fworst - fopt)) if ok.any() else None,
                p_opt=float(w[ok & (f == fopt)].sum() / tot), mean_f=float((w * f).sum() / tot))


def classical_sector_samples(P, q, shots, rng):
    w = sector_weights(P, q); keys = list(w); pr = np.array([w[k] for k in keys]); pr /= pr.sum()
    X = np.zeros((shots, P.nq), np.int8)
    for t in range(P.T):
        idx = rng.choice(len(keys), size=shots, p=pr)
        for s, j in enumerate(idx):
            L, S = keys[j]
            for d, k in ((0, L), (1, S)):
                for i in rng.choice(P.na, size=k, replace=False): X[s, P.q(t, d, i)] = 1
    return X, np.ones(shots, int)


def hw(qc):
    if qc.num_qubits > BK.num_qubits: return None
    t = transpile(qc, BK, optimization_level=3, seed_transpiler=1)
    return dict(cz=int(t.count_ops().get('cz', 0)), depth=int(t.depth()))


if __name__ == '__main__':
    keys = _sys.argv[1:] or list(INST)
    fn = 'results/portfolio_scaling_mps.json'
    out = json.load(open(fn)) if _os.path.exists(fn) else {}
    for key in keys:
        t0 = time.time(); P = load_big(key); rng = np.random.default_rng(0)
        fopt, xopt = optimum(P); fworst, _ = optimum(P, 'max')
        sc = 1 / float(np.abs(P.h).max()); D = float(np.abs(P.h).max())
        R = dict(qubits=P.nq, periods=P.T, assets=P.na, f_opt=fopt, f_worst=fworst,
                 qubits_slack_qubo=P.nq + P.T * (P.cs1 + P.cs2), opt_solution=P.solution_lines(xopt))
        X = rng.integers(0, 2, (2000, P.nq)).astype(np.int8); R['uniform'] = metrics(P, X, np.ones(2000), fopt, fworst)
        best = None
        for q in (0.1, 0.2, 0.3, 0.4, 0.5):
            X, w = classical_sector_samples(P, q, 300, rng); m = metrics(P, X, w, fopt, fworst)
            if best is None or m['mean_f'] < best[1]['mean_f']: best = (q, m)
        X, w = classical_sector_samples(P, best[0], 2000, rng)
        R['A_q classical'] = dict(q=best[0], **metrics(P, X, w, fopt, fworst))
        print(key, P.nq, 'opt', fopt, 'uniform', R['uniform'], 'A_q classical', R['A_q classical'], flush=True)
        # ---- unbalanced penalty QAOA p=1
        hu, Ju, cu = P.unbalanced_poly(D / 16, D / 16); Jus = {k: v * sc for k, v in Ju.items()}
        pen = lambda X: P.energy(X) + 0  # placeholder replaced below
        def pen_energy(Xb):
            e = cu + Xb @ hu
            for (a, b), v in Ju.items(): e = e + v * (Xb[:, a] & Xb[:, b])
            return e
        grid = []; tg = time.time()
        G = ((0.75, 1.0, 1.35, 1.8), (-0.9, -0.7, -0.5)) if P.nq <= 140 else ((1.0, 1.35), (-0.7, -0.5))   # MPS cost
        for g, b in itertools.product(*G):   # around the transferred optimum
            Xb, w = sample(qaoa_circuit(P, 'penalty_unbal', ([g], [b]), h=hu * sc, J=Jus), 300, 1)
            grid.append(((g, b), float((w * pen_energy(Xb[:, :P.nq].astype(np.int64))).sum() / w.sum())))
        (g, b), _ = min(grid, key=lambda z: z[1])
        qc = qaoa_circuit(P, 'penalty_unbal', ([g], [b]), h=hu * sc, J=Jus)
        Xb, w = sample(qc, 2000, 7)
        R['unbalanced p=1'] = dict(gamma=g, beta=b, grid_seconds=time.time() - tg, hardware=hw(qc), **metrics(P, Xb, w, fopt, fworst))
        print(key, 'unbalanced', {k: v for k, v in R['unbalanced p=1'].items()}, flush=True)
        # ---- hybrid p=1: coherent sector + Dicke, linear phase, XY path
        grid = []; tg = time.time()
        for q, g, b in itertools.product((0.25, 0.35, 0.45), (1.2, 1.8, 2.5, 3.2), (-0.95, -0.8, -0.65)):
            Xb, w = sample(sector_hybrid(P, q, [g], [b], P.h * sc), 300, 1)
            grid.append(((q, g, b), float((w * P.energy(Xb[:, :P.nq])).sum() / w.sum())))
        (q, g, b), _ = min(grid, key=lambda z: z[1])
        qc = sector_hybrid(P, q, [g], [b], P.h * sc)
        Xb, w = sample(qc, 2000, 7)
        R['hybrid p=1'] = dict(q=q, gamma=g, beta=b, grid_seconds=time.time() - tg, hardware=hw(qc), **metrics(P, Xb, w, fopt, fworst))
        # hardware cost of the per-shot version (counts drawn classically, Dicke states for those counts): mean of 5 draws
        if P.nq <= BK.num_qubits:
            wts = sector_weights(P, q); ks = list(wts); pr = np.array([wts[k] for k in ks]); pr /= pr.sum(); czs = []
            for r in range(5):
                counts = [ks[j] for j in rng.choice(len(ks), size=P.T, p=pr)]
                czs.append(hw(sector_circuit(P, counts, [g], [b], P.h * sc, 'dicke')))
            R['hybrid p=1']['hardware_per_shot_version'] = dict(cz=float(np.mean([c['cz'] for c in czs])),
                                                                depth=float(np.mean([c['depth'] for c in czs])))
        R['seconds'] = time.time() - t0
        print(key, 'hybrid', {k: v for k, v in R['hybrid p=1'].items()}, flush=True)
        out[key] = R; json.dump(out, open(fn, 'w'), indent=1)
