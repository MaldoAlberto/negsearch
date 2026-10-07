"""Our feasible preparation + pair-exchange mixers WITHOUT the two-qubit objective layer (the objective would be applied with single-qubit layers, as in Fourier-LCU, or sampled):
CZ / depth / ESP of that layer on the three cores, against the full layer.  A diagonal layer of any kind keeps the support of the state, so feasibility is unaffected.
Writes results/hybrid_nocost_<tag>.json."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)
import json, os, warnings; warnings.filterwarnings('ignore')
from qiskit import QuantumCircuit
import experiments.case_four as C
from negsearch.hw_large import swap_pairs, pair_swap_mixer
from negsearch.symlower import symmetric_block_prep
from experiments.qaoa_realistic import best as compile_best
cp = C.core_problem(); bl = cp['bl']; n = C.n; out = {}
for brick in (False, True):
    qc = QuantumCircuit(n)
    for b in bl: qc.compose(symmetric_block_prep(b['size'], b['l'], b['s']), qubits=b['xq'], inplace=True)
    for b in bl:
        if len(b['F']) == 1: continue
        xq = b['xq']; two = b['l'] == 0 or b['s'] == 0; d = 1 if b['l'] == 0 else 0; pr = swap_pairs(b['size'], 'path')
        if brick: pr = pr[::2] + pr[1::2]
        for j, k in pr:
            if two: qc.rxx(0.6, xq[2 * j + d], xq[2 * k + d]); qc.ryy(0.6, xq[2 * j + d], xq[2 * k + d]); qc.rzz(0.6, xq[2 * j + d], xq[2 * k + d])
            else: pair_swap_mixer(qc, xq[2 * j], xq[2 * j + 1], xq[2 * k], xq[2 * k + 1], 0.6)
    for q in range(n):
        if q in cp['col'] or True: pass
    for v in cp['free']: qc.rz(0.3, v)             # single-qubit objective layer (Fourier-LCU style)
    r = compile_best(C.compact(qc)[0], seeds=3); out['brick' if brick else 'sequential'] = r
    print('brick' if brick else 'sequential', r, flush=True)
json.dump(out, open(f"results/hybrid_nocost_{os.environ.get('CF_TAG', 'n64')}.json", 'w'), indent=1)
