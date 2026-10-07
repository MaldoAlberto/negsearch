"""Dicke-state preparation |D^n_k> (uniform superposition of all n-bit strings of Hamming weight k), deterministic
split-and-cyclic-shift construction of Baertschi & Eidenbenz (2019), O(kn) CNOTs, linear depth."""
import math
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate


def _scs(qc, q, n, k):
    """SCS_{n,k} on the first n qubits of list q."""
    qc.cx(q[n - 2], q[n - 1])
    qc.append(RYGate(2 * math.acos(math.sqrt(1 / n))).control(1), [q[n - 1], q[n - 2]])
    qc.cx(q[n - 2], q[n - 1])
    for l in range(2, k + 1):
        qc.cx(q[n - l - 1], q[n - 1])
        qc.append(RYGate(2 * math.acos(math.sqrt(l / n))).control(2), [q[n - 1], q[n - l], q[n - l - 1]])
        qc.cx(q[n - l - 1], q[n - 1])


def dicke(qc, q, k):
    """Append |D^n_k> preparation (from |0...0>) on qubit list q (n = len(q))."""
    n = len(q)
    if k == 0: return
    if k == n:
        for x in q: qc.x(x)
        return
    for x in q[n - k:]: qc.x(x)
    for m in range(n, k, -1): _scs(qc, q, m, k)
    for m in range(k, 1, -1): _scs(qc, q, m, m - 1)


def dicke_unitary(qc, q, k):
    """U_{n,k} of Baertschi & Eidenbenz: maps |0^{n-l} 1^l> (ones on the last l qubits of q) to |D^n_l> for every
    l <= k.  Combined with a weighted 'thermometer' input it prepares any symmetric state sum_l c_l |D^n_l>."""
    n = len(q)
    if k <= 0 or n < 2: return
    k = min(k, n - 1) if k < n else n - 1
    for m in range(n, k, -1): _scs(qc, q, m, k)
    for m in range(k, 1, -1): _scs(qc, q, m, m - 1)


def dicke_circuit(n, k):
    """standalone circuit of |D^n_k> on n qubits (used by the depth studies)"""
    qc = QuantumCircuit(n)
    dicke(qc, list(range(n)), k)
    return qc
