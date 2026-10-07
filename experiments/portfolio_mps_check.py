"""Truncation check for the MPS scaling study: same circuits with bond dimension 64 (used) vs 256, and vs the exact
statevector on the 12-qubit instance."""
import os as _os, sys as _sys
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT); _sys.path.insert(0, ROOT)
import json, numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from negsearch.portfolio_circuits import qaoa_circuit
from negsearch.portfolio_sector import sector_hybrid
from negsearch.portfolio_dp import optimum
from experiments.portfolio_scaling_mps import load_big, metrics, BASIS

R = json.load(open('results/portfolio_scaling_mps.json')); out = {}
for key in ('a003_t02', 'a005_t04', 'a010_t04'):
    P = load_big(key); r = R[key]; sc = 1 / float(np.abs(P.h).max()); D = float(np.abs(P.h).max())
    hu, Ju, cu = P.unbalanced_poly(D / 16, D / 16)
    u, hy = r['unbalanced p=1'], r['hybrid p=1']
    circs = {'unbalanced p=1': qaoa_circuit(P, 'penalty_unbal', ([u['gamma']], [u['beta']]), h=hu * sc, J={k: v * sc for k, v in Ju.items()}),
             'hybrid p=1': sector_hybrid(P, hy['q'], [hy['gamma']], [hy['beta']], P.h * sc)}
    out[key] = {}
    for name, qc in circs.items():
        qc = qc.copy(); qc.measure_all(); t = transpile(qc, basis_gates=BASIS, optimization_level=1)
        sims = [('bond64', AerSimulator(method='matrix_product_state', matrix_product_state_max_bond_dimension=64,
                                        matrix_product_state_truncation_threshold=1e-8)),
                ('bond256', AerSimulator(method='matrix_product_state', matrix_product_state_max_bond_dimension=256,
                                         matrix_product_state_truncation_threshold=1e-10))]
        if P.nq <= 20: sims.append(('statevector', AerSimulator(method='statevector')))
        for label, sim in sims:
            c = sim.run(t, shots=4000, seed_simulator=3).result().get_counts()
            bits = np.array([[int(ch) for ch in k.replace(' ', '')[::-1]] for k in c], np.int8); w = np.array(list(c.values()))
            m = metrics(P, bits, w, r['f_opt'], r['f_worst'])
            out[key][f'{name} | {label}'] = {k: m[k] for k in ('p_feas', 'quality', 'best_gap')}
            print(key, name, label, out[key][f'{name} | {label}'], flush=True)
json.dump(out, open('results/portfolio_mps_check.json', 'w'), indent=1)
