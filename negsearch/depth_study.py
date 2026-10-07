"""Circuits for the depth / viability study: Grover iterations and Grover-mixer QAOA on A_q, written with hardware-friendly pieces
(objective as a Z-polynomial from the Walsh expansion, oracle as multi-controlled Z with a Toffoli V-chain, reflections with V-chains)."""
from __future__ import annotations

import math

import numpy as np
from qiskit import QuantumCircuit

from .automaton import compile_Aq

MCZ_MODE = 'v-chain'      # 'v-chain' | 'v-chain-dirty' | 'recursion' | 'noancilla'


def mcz_vchain(qc, qubits, anc):
    """phase flip on |1..1> of `qubits`; implementation selected by MCZ_MODE (all verified equal as unitaries up to relative-phase options)"""
    qubits = list(qubits)
    if len(qubits) == 1:
        qc.z(qubits[0]); return
    if len(qubits) == 2:
        qc.cz(*qubits); return
    last = qubits[-1]
    qc.h(last)
    if MCZ_MODE == 'v-chain':
        qc.mcx(qubits[:-1], last, list(anc)[:max(0, len(qubits) - 3)], mode='v-chain')
    elif MCZ_MODE == 'v-chain-dirty':
        c = qubits[:-1]
        qc.mcx(c, last, list(anc)[:max(0, len(c) - 2)], mode='v-chain-dirty') if len(c) > 2 else qc.mcx(c, last)
    elif MCZ_MODE == 'recursion':
        qc.mcx(qubits[:-1], last, list(anc)[:1], mode='recursion')
    else:
        qc.mcx(qubits[:-1], last, mode='noancilla')
    qc.h(last)


def walsh(f, n, tol=1e-9):
    """coefficients c_S of f(x) = sum_S c_S prod_{i in S} z_i, z_i = 1 - 2 x_i; returns {tuple(qubits): c}.  f indexed MSB-first."""
    F = np.array(f, float).copy()
    h = 1
    while h < len(F):
        for i in range(0, len(F), 2 * h):
            a = F[i:i + h].copy(); b = F[i + h:i + 2 * h].copy()
            F[i:i + h] = a + b; F[i + h:i + 2 * h] = a - b
        h *= 2
    F /= len(F)
    out = {}
    for S in range(1, len(F)):
        if abs(F[S]) > tol:
            out[tuple(k for k in range(n) if (S >> (n - 1 - k)) & 1)] = float(F[S])
    return out


def phase_poly(qc, coeffs, gamma):
    """exp(-i gamma sum_S c_S Z_S) with CX ladders (qubit k = variable k)"""
    for S, c in coeffs.items():
        if len(S) == 1:
            qc.rz(2 * gamma * c, S[0])
        else:
            for a, b in zip(S[:-1], S[1:]):
                qc.cx(a, b)
            qc.rz(2 * gamma * c, S[-1])
            for a, b in reversed(list(zip(S[:-1], S[1:]))):
                qc.cx(a, b)


def mark_strings(qc, n, idxs, anc):
    """phase -1 on the listed basis strings of the first n qubits (MSB-first indices) with V-chain multi-controlled Z"""
    for idx in idxs:
        zeros = [k for k in range(n) if not (idx >> (n - 1 - k)) & 1]
        for k in zeros:
            qc.x(k)
        mcz_vchain(qc, range(n), anc)
        for k in zeros:
            qc.x(k)


def reflect_zero(qc, N, anc, beta):
    """e^{-i beta |0..0><0..0|} on qubits 0..N-1 (global phase aside); beta = pi is a plain multi-controlled Z"""
    for k in range(N):
        qc.x(k)
    if abs(beta - math.pi) < 1e-12:
        mcz_vchain(qc, range(N), anc)
    else:
        anc = list(anc)
        controls = list(range(N - 1)); tgt = anc[-1]; chain = anc[:-1]
        qc.mcx(controls, tgt, chain[:max(0, len(controls) - 2)], mode='v-chain')
        qc.cp(-beta, tgt, N - 1)
        qc.mcx(controls, tgt, chain[:max(0, len(controls) - 2)], mode='v-chain')
    for k in range(N):
        qc.x(k)


def n_ancillas(N):
    return max(0, N - 2)


def compile_Aq_onehot(A, q=0.5):
    """A_q with a ONE-HOT automaton register: level k>=1 with width w>1 gets w qubits, h_{k,s}=1 iff the state is s.
    Every operation then has at most two controls (Toffoli) instead of up to log2(w)+1.  Qubits: x_0..x_{n-1}, then registers."""
    from qiskit.circuit.library import RYGate, XGate
    n = A.n; theta = 2 * math.asin(math.sqrt(q))
    reg = {}; pos = n
    for k in range(1, n):
        if A.width[k] > 1:
            reg[k] = list(range(pos, pos + A.width[k])); pos += A.width[k]
    qc = QuantumCircuit(pos)

    def mcg(base, ctrls, states, target):
        ng = [c for c, st in zip(ctrls, states) if st == 0]
        for c in ng:
            qc.x(c)
        qc.append(base.control(len(ctrls)), ctrls + [target]) if ctrls else qc.append(base, [target])
        for c in ng:
            qc.x(c)

    for k in range(n):
        for s in range(A.width[k]):
            t0, t1 = A.trans[k][s]
            h = [reg[k][s]] if k in reg else []
            if t0 >= 0 and t1 >= 0:
                mcg(RYGate(theta), h, [1] * len(h), k)
            elif t1 >= 0:
                mcg(XGate(), h, [1] * len(h), k)
        if k + 1 in reg:
            for s in range(A.width[k]):
                for b in (0, 1):
                    t = A.trans[k][s][b]
                    if t >= 0:
                        h = [reg[k][s]] if k in reg else []
                        mcg(XGate(), h + [k], [1] * len(h) + [b], reg[k + 1][t])
    return qc, None


def grover_from_prep(Aq, n, good_idxs, k):
    N = Aq.num_qubits
    qc = QuantumCircuit(N + n_ancillas(N)); anc = list(range(N, qc.num_qubits))
    qc.compose(Aq, qubits=range(N), inplace=True)
    for _ in range(k):
        mark_strings(qc, n, good_idxs, anc)
        qc.compose(Aq.inverse(), qubits=range(N), inplace=True)
        reflect_zero(qc, N, anc, math.pi)
        qc.compose(Aq, qubits=range(N), inplace=True)
    return qc


def qaoa_from_prep(Aq, n, coeffs, params):
    N = Aq.num_qubits; p = len(params) // 2
    qc = QuantumCircuit(N + n_ancillas(N)); anc = list(range(N, qc.num_qubits))
    qc.compose(Aq, qubits=range(N), inplace=True)
    for l in range(p):
        phase_poly(qc, coeffs, params[l])
        qc.compose(Aq.inverse(), qubits=range(N), inplace=True)
        reflect_zero(qc, N, anc, params[p + l])
        qc.compose(Aq, qubits=range(N), inplace=True)
    return qc


def grover_circuit(A, good_idxs, k, q=0.5):
    return grover_from_prep(compile_Aq(A, q)[0], A.n, good_idxs, k)


def qaoa_circuit(A, coeffs, params, q=0.5):
    return qaoa_from_prep(compile_Aq(A, q)[0], A.n, coeffs, params)
