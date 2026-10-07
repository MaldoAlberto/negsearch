"""A realistic multi-constraint case: multi-period long/short portfolio with
  per period: 0 <= L-S <= Dmax, L+S <= Bmax (capital and budget), per-sector cap on long positions, no long and short on the same asset,
  and ONE global trading budget  sum_t (L_t+S_t) <= Q.
Variables: for period t, asset i (sorted by sector), direction d (0 long, 1 short).  The constraint automaton is written from the structure
(build_structural) for any variable order, so the generic compiler applies at n in the hundreds."""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from qiskit import QuantumCircuit, transpile

from .automaton import compile_Aq
from .structural import build_structural, count_feasible


def variables(A, T, order='period'):
    if order == 'period':
        return [(t, i, d) for t in range(T) for i in range(A) for d in (0, 1)]
    if order == 'asset':
        return [(t, i, d) for i in range(A) for t in range(T) for d in (0, 1)]
    raise ValueError(order)


def case_automaton(A, T, sectors, cap=1, Dmax=3, Bmax=3, Q=None, order='period'):
    """sectors: list of sector sizes summing to A.  Returns (automaton, var list)."""
    assert sum(sectors) == A
    sec_start = set(np.cumsum([0] + list(sectors))[:-1].tolist())
    var = variables(A, T, order)
    n = len(var)
    last = {}
    for k, (t, i, d) in enumerate(var):
        last[t] = k
    Q = Q if Q is not None else T * Bmax

    def step(k, s, b):
        tot, per = s
        t, i, d = var[k]
        L, S, sl, pl = per[t]
        if i in sec_start and d == 0:
            sl = 0
        if d == 0:
            L += b; sl += b
            if sl > cap:
                return None
            pl = b
        else:
            if b and pl:
                return None
            S += b; pl = 0
        tot += b
        if L + S > Bmax or tot > Q:
            return None
        per = list(per)
        if k == last[t]:
            if not (0 <= L - S <= Dmax):
                return None
            per[t] = None
        else:
            per[t] = (L, S, sl, pl)
        return (tot, tuple(per))

    init = (0, tuple((0, 0, 0, 0) for _ in range(T)))
    return build_structural(n, init, step, lambda s: True), var


def block_automaton(A, sectors, m, cap=1, Dmax=3, Bmax=3):
    """one period conditioned on exactly m positions (interface value)."""
    sec_start = set(np.cumsum([0] + list(sectors))[:-1].tolist())
    n = 2 * A

    def step(k, s, b):
        L, S, sl, pl = s
        i, d = divmod(k, 2)
        if i in sec_start and d == 0:
            sl = 0
        if d == 0:
            L += b; sl += b
            if sl > cap:
                return None
            pl = b
        else:
            if b and pl:
                return None
            S += b; pl = 0
        if L + S > m:
            return None
        return (L, S, sl, pl)
    return build_structural(n, (0, 0, 0, 0), step, lambda s: s[0] + s[1] == m and 0 <= s[0] - s[1] <= Dmax)


def feasible_bruteforce(A, T, sectors, cap, Dmax, Bmax, Q, order='period'):
    """brute-force set of feasible strings (for validation, small n)"""
    var = variables(A, T, order); n = len(var)
    sec_start = set(np.cumsum([0] + list(sectors))[:-1].tolist())
    sec_of = {}
    for s, st in enumerate(sorted(sec_start)):
        pass
    sector_id = np.repeat(np.arange(len(sectors)), sectors)
    out = set()
    for x in range(2 ** n):
        bits = [(x >> (n - 1 - k)) & 1 for k in range(n)]
        val = {v: b for v, b in zip(var, bits)}
        ok = True; tot = 0
        for t in range(T):
            L = sum(val[(t, i, 0)] for i in range(A)); S = sum(val[(t, i, 1)] for i in range(A))
            tot += L + S
            if not (0 <= L - S <= Dmax and L + S <= Bmax):
                ok = False; break
            if any(val[(t, i, 0)] and val[(t, i, 1)] for i in range(A)):
                ok = False; break
            for s in range(len(sectors)):
                if sum(val[(t, i, 0)] for i in range(A) if sector_id[i] == s) > cap:
                    ok = False; break
            if not ok:
                break
        if ok and tot <= Q:
            out.add(x)
    return out


def enumerate_automaton(Aut):
    """all accepted strings (ints, MSB first), for small n"""
    out = []

    def rec(k, s, x):
        if k == Aut.n:
            out.append(x); return
        for b in (0, 1):
            t = Aut.trans[k][s][b]
            if t >= 0:
                rec(k + 1, t, (x << 1) | b)
    rec(0, 0, 0)
    return set(out)


# ---- gate accounting -----------------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def cz_of_gate(base, nctrl):
    from qiskit.circuit.library import RYGate, XGate
    if nctrl == 0:
        return 0
    g = (RYGate(0.7) if base == 'ry' else XGate()).control(nctrl)
    qc = QuantumCircuit(nctrl + 1); qc.append(g, range(nctrl + 1))
    t = transpile(qc, basis_gates=['cz', 'rz', 'sx', 'x', 'ry'], optimization_level=1)
    return t.count_ops().get('cz', 0)


def aq_cost(Aut):
    """(qubits, #multi-controlled gates, estimated CZ with default ancilla-free synthesis) of the compiled A_q (binary register)"""
    qc, info = compile_Aq(Aut, 0.5)
    cz = 0; ng = 0
    for inst in qc.data:
        op = inst.operation
        nc = getattr(op, 'num_ctrl_qubits', 0)
        base = getattr(getattr(op, 'base_gate', None), 'name', op.name)
        if nc:
            ng += 1; cz += cz_of_gate('ry' if base == 'ry' else 'x', nc)
    return qc.num_qubits, ng, cz


def sector_block_automaton(size, l, s):
    """one sector of one period conditioned on its counts (l longs, s shorts); long and short of the same asset exclusive"""
    def step(k, st, b):
        L, S, pl = st
        j, d = divmod(k, 2)
        if d == 0:
            L += b; pl = b
        else:
            if b and pl:
                return None
            S += b; pl = 0
        if L > l or S > s:
            return None
        return (L, S, pl)
    return build_structural(2 * size, (0, 0, 0), step, lambda st: st[0] == l and st[1] == s)


def count_cz(qc):
    cz = 0; ng = 0
    for inst in qc.data:
        op = inst.operation
        nc = getattr(op, 'num_ctrl_qubits', 0)
        base = getattr(getattr(op, 'base_gate', None), 'name', op.name)
        if nc:
            ng += 1; cz += cz_of_gate('ry' if base == 'ry' else 'x', nc)
    return qc.num_qubits, ng, cz


def onehot_cost(Aut):
    from .depth_study import compile_Aq_onehot
    return count_cz(compile_Aq_onehot(Aut)[0])
