"""Depth reduction study (compile only, FakeFez): binary register (baseline) vs one-hot register vs Dicke preparation (cardinality only).
One Grover iteration and one GM-QAOA layer after the state preparation.  Writes results/hardware_depth_variants.json."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)
import json, warnings; warnings.filterwarnings('ignore')
import numpy as np
from negsearch.automaton import build_automaton, compile_Aq
from negsearch.depth_study import walsh, compile_Aq_onehot, grover_from_prep, qaoa_from_prep
from negsearch.dicke import dicke_circuit
from negsearch.fez_check import measure
from negsearch import families as fam

cases = [('exact cover 6/4 (hw)', fam.hardware_instance(), None), ('cardinality 3 of 6', fam.cardinality_qp(6, 3, 0), (6, 3)),
         ('cardinality 4 of 8', fam.cardinality_qp(8, 4, 0), (8, 4)),
         ('knapsack n=6', fam.knapsack(6, 11), None), ('MIS n=6', fam.mis(6, 0.4, 22), None),
         ('set packing n=6', fam.set_packing(6, 5, 33), None), ('3-colouring 3 vert', fam.colouring(3, 55, 0.7), None),
         ('assignment 3x3', fam.assignment_qap(3, 6), None)]
out = []
print(f"{'case':20s} {'prep':8s} {'circuit':8s} {'q':>3s} {'CZ':>5s} {'depth':>6s} {'dur':>6s} {'d/T2':>5s} {'ESP':>6s}")
for name, P, dk in cases:
    n = P.n; A = build_automaton(P.feas, n); G = list(np.flatnonzero(P.good(0.0)))
    c = (P.cost - P.opt) / (P.worst - P.opt); co = walsh(c, n)
    preps = {'binary': compile_Aq(A, 0.5)[0], 'one-hot': compile_Aq_onehot(A)[0]}
    if dk:
        preps['Dicke'] = dicke_circuit(*dk)
    for pn, Aq in preps.items():
        for cn, qc in [('A only', Aq), ('Grover', grover_from_prep(Aq, n, G, 1)), ('QAOA', qaoa_from_prep(Aq, n, co, [0.7, 1.3]))]:
            m = measure(qc); m.update(case=name, prep=pn, circuit=cn); out.append(m)
            print(f"{name:20s} {pn:8s} {cn:8s} {m['qubits']:3.0f} {m['cz']:5.0f} {m['depth']:6.0f} {m['dur_us']:6.1f} {m['dur_us']/m['t2_us']:5.2f} {m['esp']:6.3f}", flush=True)
json.dump(out, open('results/hardware_depth_variants.json', 'w'), indent=1)
