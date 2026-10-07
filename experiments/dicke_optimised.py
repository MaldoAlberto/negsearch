"""Optimised Dicke block: compiled counts, exact state check, routed onto ibm_fez (FakeFez snapshot), best of 10 transpiler seeds by ESP.
Variants: ref (Qiskit generic controlled gates), 4cx (hand-decomposed ccRY, exact), 3cx (ccRY on the SCS inputs only; exact state, verified n<=12).
Writes results/dicke_optimised.json.  Counts only; nothing executed."""
import os, sys, json
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); os.chdir(ROOT); sys.path.insert(0, ROOT)
import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime.fake_provider import FakeFez
from negsearch.dicke import dicke_circuit
import negsearch.dicke_opt as D
fez = FakeFez(); LOG = ['cx', 'rz', 'sx', 'x']
def logical(qc):
    t = transpile(qc, basis_gates=LOG, optimization_level=3, seed_transpiler=1); return dict(cz=int(t.count_ops().get('cx', 0)), depth=int(t.depth()))
def twoq_depth(t):
    return int(t.depth(lambda i: i.operation.num_qubits == 2))
def routed(qc, seeds=10):
    best = None
    for s in range(seeds):
        t = transpile(qc, fez, optimization_level=3, seed_transpiler=s)
        cz = sum(v for k, v in t.count_ops().items() if k in ('cz', 'ecr', 'cx'))
        if best is None or cz < best['cz']: best = dict(cz=int(cz), depth=int(t.depth()), depth2q=twoq_depth(t))
    return best
rows = []
for n, k in [(6, 3), (8, 4), (10, 5), (12, 6), (16, 8), (24, 12)]:
    variants = {'ref': dicke_circuit(n, k), '4cx': D.dicke_opt_circuit(n, k, cheap=False), '3cx': D.dicke_opt_circuit(n, k, cheap=True)}
    r = dict(n=n, k=k)
    if n <= 12:
        ref = Statevector(variants['ref']).data
        r['fidelity'] = {v: float(abs(np.vdot(ref, Statevector(c).data)) ** 2) for v, c in variants.items()}
    for v, c in variants.items():
        r[v] = dict(logical=logical(c), routed=routed(c, 6 if n >= 16 else 10))
    rows.append(r); print(r, flush=True)
json.dump(rows, open('results/dicke_optimised.json', 'w'))
