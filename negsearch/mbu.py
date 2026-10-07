"""Measurement-based uncomputation of the one-hot automaton register (Gidney-style), with qubit reuse.
The register qubit r_t of layer k+1 equals the OR (the transitions into t are mutually exclusive) of [h_s=1 and x_k=b] over the transitions (s,b)->t, where h_s is the register of layer k.
Instead of undoing the relative-phase Toffolis (3 CZ each), measure every register qubit in the X basis; for outcome 1 the state has picked up the phase (-1)^{r_t(x)}, which is a product of
CZ(h_s, x_k) (polarity b) over the transitions into t.  Layers are processed from the last to the first, so the controls h_s still exist.  No Toffoli is needed to uncompute.
compile_Aq_onehot_mbu(A, q, conditional=True) returns the circuit; with conditional=False the corrections are applied unconditionally, which is an upper bound on the gate count used for compilation."""
import math
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit.library import RYGate, XGate, RCCXGate


def compile_Aq_onehot_mbu(A, q=0.5, conditional=False, corrections=True):
    n = A.n; theta = 2 * math.asin(math.sqrt(q)); layers = {}
    need = 0
    for k in range(1, n):
        if A.width[k] > 1: need += A.width[k]
    pos = n; reg = {}
    for k in range(1, n):
        if A.width[k] > 1: reg[k] = list(range(pos, pos + A.width[k])); pos += A.width[k]
    qc = QuantumCircuit(pos); nr = pos - n
    cr = ClassicalRegister(max(nr, 1), 'm'); qc.add_register(cr); cbit = {q_: cr[i] for i, q_ in enumerate(range(n, pos))}

    def mark(circ, ctrls, states, tgt, kind):
        ng = [c for c, st in zip(ctrls, states) if st == 0]
        for c in ng: circ.x(c)
        if kind == 'ry': circ.append(RYGate(theta).control(len(ctrls)), ctrls + [tgt]) if ctrls else circ.ry(theta, tgt)
        elif kind == 'x': circ.append(XGate().control(len(ctrls)), ctrls + [tgt]) if ctrls else circ.x(tgt)
        else: circ.append(RCCXGate(), ctrls + [tgt]) if len(ctrls) == 2 else circ.append(XGate().control(len(ctrls)), ctrls + [tgt]) if ctrls else circ.x(tgt)
        for c in ng: circ.x(c)

    trans_into = {}
    for k in range(n):
        for s in range(A.width[k]):
            t0, t1 = A.trans[k][s]; h = [reg[k][s]] if k in reg else []
            if t0 >= 0 and t1 >= 0: mark(qc, h, [1] * len(h), k, 'ry')
            elif t1 >= 0: mark(qc, h, [1] * len(h), k, 'x')
        if k + 1 in reg:
            for s in range(A.width[k]):
                for b in (0, 1):
                    t = A.trans[k][s][b]
                    if t >= 0:
                        h = [reg[k][s]] if k in reg else []
                        mark(qc, h + [k], [1] * len(h) + [b], reg[k + 1][t], 'rc')
                        trans_into.setdefault((k + 1, t), []).append((h, k, b))
    # measurement-based uncomputation, last layer first
    for k in sorted(reg, reverse=True):
        for t, rq in enumerate(reg[k]):
            qc.h(rq); qc.measure(rq, cbit[rq])
            body = QuantumCircuit(pos)
            for (h, kk, b) in trans_into.get((k, t), []):
                ctr = h + [kk]; sts = [1] * len(h) + [b]
                ng = [c for c, st in zip(ctr, sts) if st == 0]
                for c in ng: body.x(c)
                if len(ctr) == 1: body.z(ctr[0])
                else: body.cz(ctr[0], ctr[1])
                for c in ng: body.x(c)
            if not corrections: pass
            elif conditional:
                with qc.if_test((cbit[rq], 1)):
                    for inst in body.data: qc.append(inst.operation, [qc.qubits[body.find_bit(x).index] for x in inst.qubits])
            else:
                for inst in body.data: qc.append(inst.operation, [qc.qubits[body.find_bit(x).index] for x in inst.qubits])
            qc.reset(rq)
    return qc, reg
