"""One QAOA layer (p=1) of the multi-constraint case with the symmetry-aware preparation (no registers, no uncomputation) + objective phase +
pair-exchange mixers on a path; compared with the generic-A_q version of results/qaoa_realistic.json.  Compile only (ibm_fez topology).
Writes results/qaoa_symlower.json."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)
import json, warnings; warnings.filterwarnings('ignore')
import numpy as np
from qiskit import QuantumCircuit
from negsearch.hw_large import interface_tuples, qubo_cost, ising_from_qubo, prune, x_index, swap_mixer_layer
from negsearch.depth_study import phase_poly
from negsearch.symlower import symmetric_block_prep
from experiments.qaoa_realistic import best as compile_best

CASES = [(4, 1, [2, 2]), (6, 1, [3, 3]), (4, 2, [2, 2]), (6, 2, [3, 3])]


def circuit(A, T, S, iface, co, params=(0.3, 0.8)):
    nx = 2 * A * T; qc = QuantumCircuit(nx); off = np.cumsum([0] + list(S))
    blocks = []
    for t in range(T):
        for si, size in enumerate(S):
            l, s = iface[t][si]
            xq = [x_index(A, t, off[si] + j, d) for j in range(size) for d in (0, 1)]
            qc.compose(symmetric_block_prep(size, l, s), qubits=xq, inplace=True); blocks.append(xq)
    if params:
        phase_poly(qc, co, params[0])
        for xq in blocks: swap_mixer_layer(qc, xq, params[1], pairs='path')
    return qc


def main():
    old = {(r['n'], r['variant']): r for r in json.load(open('results/qaoa_realistic.json'))}
    out = []
    print(f"{'n':>3s} {'variant':28s} {'q':>3s} {'CZ':>5s} {'depth':>6s} {'ESP':>6s}   (generic A_q, registers uncomputed: CZ / ESP)")
    for A, T, S in CASES:
        n = 2 * A * T
        h, J = qubo_cost(A, T, 5); co = ising_from_qubo(h, J)
        tup, _ = interface_tuples(A, T, S); iface = max(tup, key=lambda x: x[1])[0]
        for name, c, ov in [('symmetric prep, full objective', co, 'SWAP mixers'), ('symmetric prep, prune 25%', prune(co, 0.25), 'SWAP + prune 25%'), ('preparation only', None, 'preparation only')]:
            qc = circuit(A, T, S, iface, c, params=None if c is None else (0.3, 0.8))
            r = compile_best(qc, seeds=6); r.update(n=n, variant=name)
            o = old[(n, ov)]; r['generic_cz'] = o['cz']; r['generic_esp'] = o['esp']; out.append(r)
            print(f"{n:3d} {name:28s} {r['qubits']:3d} {r['cz']:5d} {r['depth']:6d} {r['esp']:6.3f}   ({o['cz']} / {o['esp']:.3f})", flush=True)
    json.dump(out, open('results/qaoa_symlower.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
