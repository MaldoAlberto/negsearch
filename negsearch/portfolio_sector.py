"""Ancilla-free coherent 'sector + Dicke' preparation for count constraints (portfolio, ub = 1).
Per period: a weighted thermometer on the shorts block with P(S), a thermometer on the longs block conditioned on S
with P(L | S), then the Baertschi-Eidenbenz unitary U_{n,k} on each block turns thermometers into Dicke states.
The state is sum_{L,S} sqrt(w_t(L,S)) |D^n_S>|D^n_L> per period, where w_t is the count distribution of the
negation-forced preparation A_q.  Because the XY mixer and the diagonal phase never change (L, S), every
measurement statistic equals that of drawing (L, S) classically per shot (or with a mid-circuit-measured coin) and
preparing the Dicke states: one circuit, no ancillas, no mid-circuit measurement."""
import math
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate
from .dicke import dicke_unitary
from .portfolio_circuits import cost_layer, xy_edges
from qiskit.circuit.library import XXPlusYYGate


def sector_weights(P, q):
    """Distribution of the final counts (L, S) of one period under A_q (coin probability q)."""
    order, act = P.plan(); dist = {(0, 0): 1.0}
    for k, (d, i) in enumerate(order):
        nd = {}
        for (L, S), w in dist.items():
            dec = act[k][(L, S)]
            opts = ((0, 1 - q), (1, q)) if dec == 'coin' else ((dec, 1.0),)
            for v, pv in opts:
                key = (L + (d == 0) * v, S + (d == 1) * v); nd[key] = nd.get(key, 0) + w * pv
        dist = nd
    return {k: v for k, v in dist.items() if v > 1e-15}


def _ry_angle(p):
    return 2 * math.asin(math.sqrt(min(1.0, max(0.0, p))))


def sector_prep(P, q, qc=None):
    w = sector_weights(P, q); n = P.na
    qc = qc or QuantumCircuit(P.nq)
    Smax = max(S for _, S in w); Lmax = max(L for L, _ in w)
    pS = {s: sum(v for (L, S), v in w.items() if S == s) for s in range(Smax + 1)}
    for t in range(P.T):
        qs = [P.q(t, 1, i) for i in range(n)]; ql = [P.q(t, 0, i) for i in range(n)]
        sb = lambda j: qs[n - j]                 # j-th thermometer bit (1-indexed) of the shorts block
        lb = lambda j: ql[n - j]
        # shorts thermometer
        for j in range(1, Smax + 1):
            ge_prev = sum(pS[s] for s in range(j - 1, Smax + 1)); ge = sum(pS[s] for s in range(j, Smax + 1))
            th = _ry_angle(ge / ge_prev if ge_prev > 0 else 0)
            if j == 1: qc.ry(th, sb(1))
            else: qc.append(RYGate(th).control(1), [sb(j - 1), sb(j)])
        # longs thermometer conditioned on S
        for s in range(Smax + 1):
            if pS[s] <= 0: continue
            pL = {l: w.get((l, s), 0) / pS[s] for l in range(Lmax + 1)}
            cond = ([sb(s)] if s >= 1 else []) ; cstate = '1' * len(cond)
            if s + 1 <= Smax: cond = cond + [sb(s + 1)]; cstate = '0' + cstate     # ctrl_state is little-endian
            for j in range(1, Lmax + 1):
                ge_prev = sum(pL[l] for l in range(j - 1, Lmax + 1)); ge = sum(pL[l] for l in range(j, Lmax + 1))
                if ge_prev <= 0: break
                th = _ry_angle(ge / ge_prev)
                if th == 0: break
                ctr = cond + ([lb(j - 1)] if j >= 2 else []); cs = ('1' if j >= 2 else '') + cstate
                if ctr: qc.append(RYGate(th).control(len(ctr), ctrl_state=cs), ctr + [lb(j)])
                else: qc.ry(th, lb(j))
        dicke_unitary(qc, qs, Smax); dicke_unitary(qc, ql, Lmax)
    return qc


def sector_hybrid(P, q, gam, bet, h, J=None):
    qc = sector_prep(P, q)
    for g, b in zip(gam, bet):
        cost_layer(qc, list(range(P.nq)), h, J or {}, g)
        for L in xy_edges(P):
            for a, c in L: qc.append(XXPlusYYGate(2 * b), [a, c])
    return qc
