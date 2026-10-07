"""Validation of the hardware circuits before they are run (no hardware):
 (1) n=8 instance (A=4, T=1): the logical QAOA p=1 circuit (state vector) against the exact tensor-product numerics, all interface tuples;
 (2) n=24 instance: the logical preparation circuit on Aer's matrix-product-state simulator: 100% feasible samples and distance to the exact
     product distribution (total variation, shot noise included).  Writes results/hardware_large_case_validation.json."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)
import json, warnings; warnings.filterwarnings('ignore')
import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
import experiments.hardware_large_case as H
from negsearch.hw_large import interface_tuples, qubo_cost, ising_from_qubo, build_hw_circuit

res = {}
# (1) n=8 QAOA
H.A, H.T, H.SECT, H.NV = 4, 1, [2, 2], 8
h, J = qubo_cost(4, 1, H.SEED); co = ising_from_qubo(h, J)
tuples, tot = interface_tuples(4, 1, [2, 2])
params = [0.37, 0.81]
worst_err = 0.0
for iface, w in tuples:
    bl = H.tuple_support(iface); X, combos = H.full_strings(bl); c = H.cost_of(X, h, J)
    cn = (c - c.min()) / max(np.ptp(c), 1e-9) * 0 + c          # raw cost, same as the circuit's phase layer (global offsets are global phases)
    P_t = H.tensor_qaoa(bl, combos, cn, params)[tuple(combos.T)]
    qc, _ = build_hw_circuit(4, 1, [2, 2], iface, co, params, measure=False)
    p = Statevector(qc).probabilities(qargs=list(range(8)))
    idx = (X * (1 << np.arange(8))).sum(1)
    err = np.abs(p[idx] - P_t).max(); worst_err = max(worst_err, err)
    assert abs(p.sum() - p[idx].sum()) < 1e-9, 'probability outside the interface support'
res['n8_qaoa_max_abs_error'] = float(worst_err); res['n8_circuit_qubits'] = qc.num_qubits
print(f'n=8 QAOA p=1: logical circuit vs exact numerics, {len(tuples)} interface tuples, max |dP| = {worst_err:.2e}, circuit qubits {qc.num_qubits}')

# (2) n=24 preparation on MPS
H.A, H.T, H.SECT, H.NV = 6, 2, [3, 3], 24
h, J = qubo_cost(6, 2, H.SEED); co = ising_from_qubo(h, J)
tuples, tot = interface_tuples(6, 2, [3, 3])
sim = AerSimulator(method='matrix_product_state')
rows = []
for k in sorted(range(len(tuples)), key=lambda i: -tuples[i][1])[:3]:
    iface, w = tuples[k]
    qc, bl_ = build_hw_circuit(6, 2, [3, 3], iface, co, [], measure=True)
    t = transpile(qc, sim, optimization_level=0)
    shots = 20000
    counts = sim.run(t, shots=shots).result().get_counts()
    bl = H.tuple_support(iface); X, combos = H.full_strings(bl)
    exact = {}
    amps = [np.array([m for _, m in b[2]]) for b in bl]
    for row, cb in zip(X, combos):
        pr = float(np.prod([amps[b][cb[b]] for b in range(len(bl))]))
        exact[''.join(str(int(v)) for v in row[::-1])] = pr          # classical bit string: qubit 0 rightmost
    tv = 0.5 * sum(abs(counts.get(s, 0) / shots - exact.get(s, 0)) for s in set(counts) | set(exact))
    feas = sum(v for s, v in counts.items() if s in exact) / shots
    rows.append(dict(iface=str(iface), tv=tv, feasible=feas, support=len(exact)))
    print(f'n=24 prep, interface {iface}: support {len(exact)} strings, feasible samples {feas:.4f}, total variation to exact {tv:.4f} (shot noise ~{np.sqrt(len(exact)/shots)/2:.3f})', flush=True)
res['n24_prep'] = rows
json.dump(res, open('results/hardware_large_case_validation.json', 'w'), indent=1)
