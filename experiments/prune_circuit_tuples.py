"""The pruning stages over many interface tuples (not only the most probable one): CZ and ESP of S0 (baseline), S2 (exact: forced variables eliminated,
trivial blocks, two-qubit mixers) and S3 (+ constraint gauge) for interface tuples sampled with their weight.  Compile only.  Also checks, for n=24,
that S3 and S0 give the same state up to a global phase (two Qiskit statevector simulations, random angles).  Writes results/prune_circuit_tuples.json."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)
import json, warnings; warnings.filterwarnings('ignore')
import numpy as np
from qiskit.quantum_info import Statevector
from negsearch.hw_large import interface_tuples, qubo_cost
import experiments.prune_circuit as P
from experiments.qaoa_realistic import best as compile_best

rng = np.random.default_rng(11); out = []
for A, T, S in P.CASES:
    n = 2 * A * T; h0, J0 = qubo_cost(A, T, 5); tup, _ = interface_tuples(A, T, S)
    w = np.array([t[1] for t in tup]); w /= w.sum()
    pick = rng.choice(len(tup), size=min(12, len(tup)), replace=False, p=w) if len(tup) > 12 else range(len(tup))
    rows = {k: [] for k in ('S0', 'S2', 'S3')}
    for ti in pick:
        iface = tup[ti][0]; bl = P.blocks_of(A, T, S, iface); forced = P.forced_values(bl)
        partner = {}
        for b in bl:
            for j in range(b['size']): partner[b['xq'][2 * j]] = b['xq'][2 * j + 1]; partner[b['xq'][2 * j + 1]] = b['xq'][2 * j]
        h1, J1 = P.eliminate(h0, J0, forced); h3, J3 = P.gauge(h1, J1, P.groups_of(bl, forced), partner)
        for k, (hh, JJ, sm) in {'S0': (h0, J0, False), 'S2': (h1, J1, True), 'S3': (h3, J3, True)}.items():
            r = compile_best(P.circuit(A, T, S, bl, hh, JJ, [0.3], [0.8], sm), seeds=3); rows[k].append((r['cz'], r['esp']))
        if n == 24 and ti == pick[0]:
            g, b_ = rng.uniform(-2, 2, 1), rng.uniform(0, 3, 1)
            a = Statevector(P.circuit(A, T, S, bl, h0, J0, g, b_, False)).data; c = Statevector(P.circuit(A, T, S, bl, h3, J3, g, b_, True)).data
            print(f'   n=24 check: 1-|<S0|S3>| = {1 - abs(np.vdot(a, c)):.1e}', flush=True)
    rec = dict(n=n, tuples=len(rows['S0']))
    for k, v in rows.items():
        cz = np.array([x[0] for x in v]); es = np.array([x[1] for x in v]); rec[k] = dict(cz_mean=float(cz.mean()), cz_min=int(cz.min()), cz_max=int(cz.max()), esp_median=float(np.median(es)), esp_min=float(es.min()))
    out.append(rec)
    print(f"n={n:2d} ({rec['tuples']} tuples): " + '; '.join(f"{k}: CZ {rec[k]['cz_mean']:.0f} ({rec[k]['cz_min']}-{rec[k]['cz_max']}), ESP median {rec[k]['esp_median']:.2f} min {rec[k]['esp_min']:.2f}" for k in ('S0', 'S2', 'S3')), flush=True)
json.dump(out, open('results/prune_circuit_tuples.json', 'w'), indent=1)
