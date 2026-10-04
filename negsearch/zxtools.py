"""ZX-calculus simplification (PyZX full_reduce + extraction) and gate accounting."""
from __future__ import annotations

import time

import numpy as np
import pyzx as zx
from qiskit import QuantumCircuit, qasm2, transpile
from qiskit.quantum_info import Statevector

CLIFFORD_T = ["cx", "h", "t", "tdg", "s", "sdg", "x", "z"]


def compact(qc: QuantumCircuit) -> QuantumCircuit:
    used = sorted({qc.find_bit(q).index for i in qc.data for q in i.qubits})
    m = {qc.qubits[u]: k for k, u in enumerate(used)}
    new = QuantumCircuit(len(used))
    for inst in qc.data:
        if inst.operation.name in ("measure", "barrier"):
            continue
        new.append(inst.operation, [new.qubits[m[q]] for q in inst.qubits])
    return new


def to_clifford_t(qc):
    return compact(transpile(qc, basis_gates=CLIFFORD_T, optimization_level=0))


def counts_qiskit(qc):
    o = qc.count_ops()
    return dict(T=o.get("t", 0) + o.get("tdg", 0), CX=o.get("cx", 0), total=sum(o.values()),
                qubits=qc.num_qubits)


def counts_zx(c):
    two = sum(1 for g in c.gates if g.name in ("CNOT", "CZ"))
    return dict(T=c.tcount(), CX=two, total=len(c.gates), qubits=c.qubits)


def zx_reduce(qc_ct: QuantumCircuit):
    t0 = time.time()
    c = zx.Circuit.from_qasm(qasm2.dumps(qc_ct))
    g = c.to_graph()
    zx.simplify.full_reduce(g)
    c2 = zx.extract_circuit(g.copy()).to_basic_gates()
    c2 = zx.optimize.basic_optimization(c2)
    return c2, time.time() - t0


def zx_to_qiskit(c):
    return QuantumCircuit.from_qasm_str(c.to_qasm())


def equivalent_on_inputs(qa, qb, n_vars, rng, trials=6):
    """Compare |<psi_a|psi_b>|^2 on random variable-register inputs (ancillas |0>), with a
    Hadamard on the first variable to expose relative phases."""
    worst = 1.0
    for _ in range(trials):
        prep = QuantumCircuit(qa.num_qubits)
        for i in range(n_vars):
            if rng.random() < 0.5:
                prep.x(i)
        prep.h(0)
        s1 = Statevector(prep.compose(qa))
        s2 = Statevector(prep.compose(qb))
        worst = min(worst, abs(np.vdot(s1.data, s2.data)) ** 2)
    return worst
