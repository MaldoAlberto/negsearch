"""IDEAL vs OPTIMISED vs EXECUTABLE QAOA cost for the conditioned multi-constraint case (p=1), compile only on the ibm_fez topology.
Variants of the circuit for the same interface tuple (the most probable one):
  GM    : block Grover mixers (A^dagger, reflection, A)                       [first version]
  SWAP  : preparation with uncomputed registers + objective phase + pair-swap block mixers (this work)
  SWAP+prune: same, objective couplings with |coefficient| below a fraction of the largest dropped (feasibility is unaffected)
Reports CZ, depth, qubits, estimated success probability (ESP), duration/T2.  Writes results/qaoa_realistic.json."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)
import json, warnings; warnings.filterwarnings('ignore')
import numpy as np
from qiskit import QuantumCircuit, transpile
from negsearch.hw_large import (interface_tuples, qubo_cost, ising_from_qubo, build_hw_circuit, build_hw_circuit_swap, prune)
from negsearch.depth_study import phase_poly
from negsearch.fez_check import bk, esp_and_time

CASES = [(4, 1, [2, 2]), (6, 1, [3, 3]), (4, 2, [2, 2]), (6, 2, [3, 3])]


def best(qc, seeds=8):
    b = None
    for s in range(seeds):
        t = transpile(qc, bk, optimization_level=3, seed_transpiler=s + 1)
        e = esp_and_time(t)[0]
        if b is None or e > b[0]:
            b = (e, t)
    e, t = b
    sch = transpile(t, bk, optimization_level=0, scheduling_method='alap')
    used = sorted({t.find_bit(q).index for i in t.data for q in i.qubits})
    t2 = float(np.median([bk.qubit_properties(q).t2 for q in used]))
    return dict(cz=t.count_ops().get('cz', 0), depth=t.depth(), qubits=len(used), esp=e, dur_t2=sch.duration * bk.dt / t2)


def main():
    out = []
    print(f"{'case':14s} {'variant':22s} {'q':>3s} {'CZ':>5s} {'depth':>6s} {'d/T2':>5s} {'ESP':>6s}")
    for A, T, S in CASES:
        h, J = qubo_cost(A, T, 5); co = ising_from_qubo(h, J)
        tup, tot = interface_tuples(A, T, S); iface = max(tup, key=lambda x: x[1])[0]
        nx = 2 * A * T
        variants = {
            'GM (first version)': build_hw_circuit(A, T, S, iface, co, [0.3, 0.8], False)[0],
            'SWAP mixers': build_hw_circuit_swap(A, T, S, iface, co, [0.3, 0.8], False)[0],
            'SWAP + prune 10%': build_hw_circuit_swap(A, T, S, iface, prune(co, 0.10), [0.3, 0.8], False)[0],
            'SWAP + prune 25%': build_hw_circuit_swap(A, T, S, iface, prune(co, 0.25), [0.3, 0.8], False)[0],
            'preparation only': build_hw_circuit_swap(A, T, S, iface, co, [], False)[0],
        }
        for k, qc in variants.items():
            r = best(qc); r.update(case=f'A={A},T={T} (n={nx})', variant=k, n=nx, terms=len([1 for c in (prune(co, 0.10) if '10' in k else prune(co, 0.25) if '25' in k else co) if len(c) == 2]))
            out.append(r)
            print(f"{r['case']:14s} {k:22s} {r['qubits']:3d} {r['cz']:5d} {r['depth']:6d} {r['dur_t2']:5.2f} {r['esp']:6.3f}", flush=True)
    json.dump(out, open('results/qaoa_realistic.json', 'w'), indent=1)



if __name__ == '__main__':
    main()
