# Hardware-ready Qiskit circuits for constraint-guided QAOA (and penalty / Hadfield baselines).
"""Qiskit circuits (hardware-ready) for constraint-guided QAOA on MIS."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate, PhaseGate, RXGate

def prep_circuit(n, N, order, q, inverse=False):
    th = 2*np.arcsin(np.sqrt(q)); pos = {v: i for i, v in enumerate(order)}
    qc = QuantumCircuit(n)
    for v in (list(order)[::-1] if inverse else order):
        e = [u for u in N[v] if pos[u] < pos[v]]
        ang = -th if inverse else th
        if e: qc.append(RYGate(ang).control(len(e), ctrl_state=0), e + [v])
        else: qc.ry(ang, v)
    return qc

def cg_qaoa(n, N, order, q, gammas, betas, measure=True):
    qc = QuantumCircuit(n)
    A = prep_circuit(n, N, order, q); Ad = prep_circuit(n, N, order, q, inverse=True)
    qc.compose(A, inplace=True)
    for g, b in zip(gammas, betas):
        for v in range(n): qc.p(g, v)          # e^{-i g C}, C = -|x|  ->  phase +g per selected vertex
        qc.compose(Ad, inplace=True)
        # e^{-i b |0><0|}: phase -b on |0...0>
        qc.x(range(n)); qc.append(PhaseGate(-b).control(n-1), list(range(n))); qc.x(range(n))
        qc.compose(A, inplace=True)
    if measure: qc.measure_all()
    return qc

def penalty_qaoa(n, E, lam, gammas, betas, measure=True):
    qc = QuantumCircuit(n); qc.h(range(n))
    deg = [0]*n
    for a, b in E: deg[a] += 1; deg[b] += 1
    for g, b in zip(gammas, betas):
        # C = -sum x_i + lam sum x_a x_b ; x = (1-Z)/2
        for v in range(n): qc.rz(2*g*(0.5 - lam*deg[v]/4), v)       # linear Z terms (global phase dropped)
        for a, b2 in E: qc.rzz(2*g*lam/4, a, b2)
        qc.rx(2*b, range(n))
    if measure: qc.measure_all()
    return qc

def hadfield_qaoa(n, N, gammas, betas, measure=True):
    qc = QuantumCircuit(n)
    for g, b in zip(gammas, betas):
        for v in range(n): qc.p(g, v)
        for v in range(n):
            e = sorted(N[v])
            qc.append(RXGate(2*b).control(len(e), ctrl_state=0), e + [v]) if e else qc.rx(2*b, v)
    if measure: qc.measure_all()
    return qc


# ---------------- ancilla-assisted (hardware-efficient) versions ----------------
def _or_zero_chain(qc, ctrls, anc, tgt):
    """tgt ^= [all ctrls == 0] using relative-phase Toffolis on a chain of ancillas (compute only)."""
    qc.x(ctrls)
    if len(ctrls) == 1:
        qc.cx(ctrls[0], tgt); ops = []
    else:
        ops = []; prev = ctrls[0]
        for i, c in enumerate(ctrls[1:-1]):
            qc.rccx(prev, c, anc[i]); ops.append((prev, c, anc[i])); prev = anc[i]
        qc.ccx(prev, ctrls[-1], tgt)
        for p_, c, a in reversed(ops): qc.rccx(p_, c, a)      # rccx is self-inverse up to the relative phase it undoes
    qc.x(ctrls)

def prep_circuit_anc(n, N, order, q, inverse=False, n_anc=None):
    """Same unitary as prep_circuit on the n variable qubits; ancillas start and end in |0>."""
    th = 2*np.arcsin(np.sqrt(q)); pos = {v: i for i, v in enumerate(order)}
    kmax = max([len([u for u in N[v] if pos[u] < pos[v]]) for v in range(n)] + [1])
    n_anc = n_anc or (kmax - 2 + 1 if kmax > 1 else 1)
    qc = QuantumCircuit(n + n_anc)
    flag = n + n_anc - 1; chain = list(range(n, n + n_anc - 1))
    for v in (list(order)[::-1] if inverse else order):
        e = sorted(u for u in N[v] if pos[u] < pos[v])
        ang = -th if inverse else th
        if not e: qc.ry(ang, v); continue
        _or_zero_chain(qc, e, chain, flag)
        qc.cry(ang, flag, v)
        _or_zero_chain(qc, e, chain, flag)
    return qc

def mc_zero_phase(qc, qubits, anc, flag, phi):
    """phase e^{i phi} on |0...0> of `qubits` via the ancilla chain."""
    _or_zero_chain(qc, qubits[:-1], anc, flag)
    qc.x(qubits[-1]); qc.cp(phi, flag, qubits[-1]); qc.x(qubits[-1])
    _or_zero_chain(qc, qubits[:-1], anc, flag)

def cg_qaoa_anc(n, N, order, q, gammas, betas, measure=True):
    A = prep_circuit_anc(n, N, order, q)
    n_anc = max(A.num_qubits - n, n - 1)          # chain long enough for the n-qubit reflection
    A = prep_circuit_anc(n, N, order, q, n_anc=n_anc); Ad = prep_circuit_anc(n, N, order, q, inverse=True, n_anc=n_anc)
    qc = QuantumCircuit(n + n_anc, n)
    flag = n + n_anc - 1; chain = list(range(n, n + n_anc - 1))
    qc.compose(A, inplace=True)
    for g, b in zip(gammas, betas):
        for v in range(n): qc.p(g, v)
        qc.compose(Ad, inplace=True)
        mc_zero_phase(qc, list(range(n)), chain, flag, -b)
        qc.compose(A, inplace=True)
    if measure: qc.measure(range(n), range(n))
    return qc
