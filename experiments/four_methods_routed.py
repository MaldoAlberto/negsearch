"""Routed (ibm_fez topology, best of 3 transpiler seeds) CZ / depth / ESP of one p=1 layer of our QAOA vs the objective-only layer shared by Fourier-LCU(single basis)
and unbalanced, on intermediate cores (about 20-45 free qubits).  All methods get the same classically reduced objective (forced variables eliminated).
Writes results/four_methods_routed.json."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)
import json, warnings; warnings.filterwarnings('ignore')
import numpy as np
from qiskit import QuantumCircuit
from negsearch.hw_large import qubo_cost, ising_from_qubo
from negsearch.depth_study import phase_poly
from negsearch.large_case import sector_block_automaton
import experiments.prune_circuit as P
from experiments.four_methods import sample
from experiments.core_map import block_free
from experiments.qaoa_realistic import best as compile_best

out = []
for fam, m, T in [('tight', 4, 2), ('medium', 4, 2), ('medium', 6, 2), ('loose', 6, 2), ('medium', 6, 3), ('loose', 8, 2)]:
    A, S = 2 * m, [m, m]; n = 2 * A * T; h0, J0 = qubo_cost(A, T, 5)
    rows = []
    for blocks in sample(fam, m, T, ntup=4, seed=3):
        iface = [tuple(blocks[t * 2:(t + 1) * 2]) for t in range(T)]
        bl = P.blocks_of(A, T, S, iface); forced = P.forced_values(bl); h1, J1 = P.eliminate(h0, J0, forced)
        free = sum(block_free(m, l, s) for l, s in blocks)
        ours = compile_best(P.circuit(A, T, S, bl, h1, J1, [0.3], [0.8], True), seeds=3)
        qc = QuantumCircuit(n); co = ising_from_qubo(h1, J1); phase_poly(qc, co, 0.3)
        for v in range(n):
            if v not in forced: qc.rx(0.8, v)
        base = compile_best(qc, seeds=3)
        rows.append(dict(free=free, ours_cz=ours['cz'], ours_depth=ours.get('depth'), ours_esp=ours['esp'], base_cz=base['cz'], base_depth=base.get('depth'), base_esp=base['esp']))
        print(fam, n, rows[-1], flush=True)
    out.append(dict(family=fam, m=m, T=T, n=n, rows=rows))
json.dump(out, open('results/four_methods_routed.json', 'w'), indent=1)
