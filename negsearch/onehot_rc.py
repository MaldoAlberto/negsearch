"""A_q with a one-hot automaton register and RELATIVE-PHASE Toffolis (3 CX instead of 6) for the register updates.
The relative phases are diagonal in (h_s, x_k, h'_t); these qubits are only used as controls afterwards, so the phases are a function of the
computational values, and the mirrored register-update sequence (uncompute=True) cancels them exactly."""
import math
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate, XGate, RCCXGate


def compile_Aq_onehot_rc(A, q=0.5, uncompute=True):
    n = A.n; theta = 2 * math.asin(math.sqrt(q))
    reg = {}; pos = n
    for k in range(1, n):
        if A.width[k] > 1:
            reg[k] = list(range(pos, pos + A.width[k])); pos += A.width[k]
    qc = QuantumCircuit(pos)
    updates = []

    def cgate(circ, base, ctrls, states, target, rc=False):
        ng = [c for c, st in zip(ctrls, states) if st == 0]
        for c in ng: circ.x(c)
        if rc and len(ctrls) == 2:
            circ.append(RCCXGate(), ctrls + [target])
        else:
            circ.append(base.control(len(ctrls)), ctrls + [target]) if ctrls else circ.append(base, [target])
        for c in ng: circ.x(c)

    for k in range(n):
        for s in range(A.width[k]):
            t0, t1 = A.trans[k][s]
            h = [reg[k][s]] if k in reg else []
            if t0 >= 0 and t1 >= 0:
                cgate(qc, RYGate(theta), h, [1] * len(h), k)
            elif t1 >= 0:
                cgate(qc, XGate(), h, [1] * len(h), k)
        if k + 1 in reg:
            blk = QuantumCircuit(pos)
            for s in range(A.width[k]):
                for b in (0, 1):
                    t = A.trans[k][s][b]
                    if t >= 0:
                        h = [reg[k][s]] if k in reg else []
                        cgate(blk, XGate(), h + [k], [1] * len(h) + [b], reg[k + 1][t], rc=True)
            qc.compose(blk, inplace=True); updates.append(blk)
    if uncompute:
        for blk in reversed(updates):
            qc.compose(blk.inverse(), inplace=True)
    return qc, reg
