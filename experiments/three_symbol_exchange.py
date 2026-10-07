"""Negative result: preparing the three-symbol (none/long/short) arrangement state with a network of adjacent unit exchanges exp(i theta SWAP_units).
Gate built in the difference basis (CX(L1,L2), CX(S1,S2), H on L1,S1, 4-qubit diagonal, undo); verified against exp(i theta P) on the 9-dimensional reachable subspace.
Reports its compiled CX count and the number of exchanges a brick-wall needs before every arrangement has nonzero support.  Counts only."""
import os, sys, math
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); os.chdir(ROOT); sys.path.insert(0, ROOT)
import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import DiagonalGate
from qiskit.quantum_info import Operator
th = 0.7
qc = QuantumCircuit(4); qc.cx(0, 2); qc.cx(1, 3); qc.h(0); qc.h(1)
vals = []
for i in range(16):
    b = [(i >> (3 - k)) & 1 for k in range(4)]; vals.append(np.exp(1j * th * (-1) ** ((b[2] & b[0]) ^ (b[3] & b[1]))))
qc.append(DiagonalGate(vals), [0, 1, 2, 3]); qc.h(0); qc.h(1); qc.cx(1, 3); qc.cx(0, 2)
P = np.zeros((16, 16))
for i in range(16):
    b = [(i >> (3 - k)) & 1 for k in range(4)]; P[(b[2] << 3) | (b[3] << 2) | (b[0] << 1) | b[1], i] = 1
U = {'N': (0, 0), 'L': (1, 0), 'S': (0, 1)}
idx = [(U[u] + U[v])[0] << 3 | (U[u] + U[v])[1] << 2 | (U[u] + U[v])[2] << 1 | (U[u] + U[v])[3] for u in 'NLS' for v in 'NLS']
print('max error on reachable subspace', np.abs(Operator(qc).data[:, idx] - expm(1j * th * P)[:, idx]).max())
print('CX per exchange', transpile(qc, basis_gates=['cx', 'rz', 'sx', 'x'], optimization_level=3, seed_transpiler=1).count_ops().get('cx'))
def layers(m, l, s):
    cur = {tuple(['L'] * l + ['S'] * s + ['N'] * (m - l - s))}; tgt = math.comb(m, l) * math.comb(m - l, s); n = 0
    while len(cur) < tgt and n < 4 * m:
        new = set(cur)
        for a in cur:
            for i in range(n % 2, m - 1, 2):
                if a[i] != a[i + 1]: b = list(a); b[i], b[i + 1] = b[i + 1], b[i]; new.add(tuple(b))
        cur = new; n += 1
    return n, sum(len(range(k % 2, m - 1, 2)) for k in range(n))
for m, l, s in [(5, 2, 2), (6, 3, 2), (8, 3, 3), (12, 4, 4)]: print((m, l, s), 'layers, exchanges', layers(m, l, s))
