"""Qiskit circuits for the QOBLIB portfolio experiments (ub = 1 model of negsearch.portfolio).

prep_circuit(P, q)      negation-forced preparation A_q: per period, two small counters (#long, #short) are
                        built on ancillas, each variable gets RY(theta) and the few counter states where the
                        constraints force it are corrected with a controlled rotation; counters are uncomputed
                        and reused for the next period.
qaoa layers             cost phase (RZ / RZZ), X mixer, XY (sector-preserving) mixer, Grover mixer A e^{-i b|0><0|} A^dag.
"""
import math, itertools
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RYGate, XXPlusYYGate


def _bits(v, w):
    return [(v >> k) & 1 for k in range(w)]


def _cover(target, same, reach, w):
    """Disjoint set of (mask, value) patterns over w counter bits (mask bit = 1 -> bit is checked) such that every
    reachable state in `target` is matched by exactly one pattern and no reachable state outside `same` is
    matched (greedy don't-care expansion; unreachable counter values are don't-cares)."""
    pats = []
    match = lambda r, m, v: (r & m) == v
    for s in sorted(target):
        if any(match(s, m, v) for m, v in pats): continue
        mask = (1 << w) - 1
        for b in range(w):                       # try to drop control bits
            m2 = mask & ~(1 << b)
            ok = all(not match(r, m2, s & m2) or (r in same and not any(match(r, m, v) for m, v in pats))
                     for r in reach)
            if ok: mask = m2
        pats.append((mask, s & mask))
    return pats


def prep_circuit(P, q, name='A'):
    order, act = P.plan()
    maxL = max(L for a in act for (L, S) in a)       # counters only ever hold prefix states listed in the plan
    maxS = max(S for a in act for (L, S) in a)
    wl, ws = max(1, math.ceil(math.log2(maxL + 1))), max(1, math.ceil(math.log2(maxS + 1)))
    x = QuantumRegister(P.nq, 'x'); cl = QuantumRegister(wl, 'cL'); cs = QuantumRegister(ws, 'cS')
    qc = QuantumCircuit(x, cl, cs, name=name)
    th = 2 * math.asin(math.sqrt(q))
    ctr = list(cl) + list(cs); w = wl + ws
    enc = lambda L, S: L | (S << wl)

    def inc(reg, ctrl):                          # reg += 1 (mod 2^len) controlled on ctrl
        n = len(reg)
        for k in reversed(range(n)):
            cs_ = [ctrl] + list(reg[:k])
            if len(cs_) == 1: qc.cx(cs_[0], reg[k])
            else: qc.mcx(cs_, reg[k])

    for t in range(P.T):
        ops = []                                  # remember increments for uncomputation
        for k, (d, i) in enumerate(order):
            tgt = x[P.q(t, d, i)]
            a = act[k]
            reach = {enc(L, S) for (L, S) in a}
            dec = {enc(L, S): v for (L, S), v in a.items()}
            n_coin = sum(v == 'coin' for v in dec.values())
            if n_coin == 0 and all(v == 0 for v in dec.values()):
                pass
            else:
                qc.ry(th, tgt)
                for val, ang in ((0, -th), (1, math.pi - th)):
                    targ = {s for s, v in dec.items() if v == val}
                    if not targ: continue
                    for mask, value in _cover(targ, targ, reach, w):
                        cbits = [b for b in range(w) if (mask >> b) & 1]
                        if not cbits:
                            qc.ry(ang, tgt); continue
                        state = ''.join(str((value >> b) & 1) for b in reversed(cbits))
                        qc.append(RYGate(ang).control(len(cbits), ctrl_state=state), [ctr[b] for b in cbits] + [tgt])
            reg = cl if d == 0 else cs
            if k < len(order) - 1:                  # counter value after the last variable is not needed
                inc(reg, tgt); ops.append((reg, tgt))
        for reg, ctrl in reversed(ops):           # uncompute counters
            n = len(reg)
            for k in range(n):
                cs_ = [ctrl] + list(reg[:k])
                if len(cs_) == 1: qc.cx(cs_[0], reg[k])
                else: qc.mcx(cs_, reg[k])
    return qc


def prep_circuit_lean(P, q, name='A'):
    """A_q for the XY hybrid: one counter pair per period (periods prepared in parallel) and NO uncomputation.
    After the last variable the counters hold (#long, #short) of their period, a function of x that the XY mixer
    conserves and the cost phase does not touch, so every computational-basis measurement statistic on x is
    identical to that of prep_circuit (the counters only label the conserved sectors).  Not valid for the
    Grover mixer, which needs A_q^dagger."""
    order, act = P.plan()
    finals = [(L, S) for L in range(P.na + 1) for S in range(P.na + 1) if P.ok_counts(L, S)]
    maxL = max([L for a in act for (L, S) in a] + [L for L, _ in finals])
    maxS = max([S for a in act for (L, S) in a] + [S for _, S in finals])
    wl, ws = max(1, math.ceil(math.log2(maxL + 1))), max(1, math.ceil(math.log2(maxS + 1)))
    x = QuantumRegister(P.nq, 'x'); regs = []
    for t in range(P.T):
        regs.append((QuantumRegister(wl, f'cL{t}'), QuantumRegister(ws, f'cS{t}')))
    qc = QuantumCircuit(x, *[r for pair in regs for r in pair], name=name)
    th = 2 * math.asin(math.sqrt(q)); w = wl + ws
    enc = lambda L, S: L | (S << wl)

    def inc(reg, ctrl):
        for k in reversed(range(len(reg))):
            cs_ = [ctrl] + list(reg[:k])
            if len(cs_) == 1: qc.cx(cs_[0], reg[k])
            else: qc.mcx(cs_, reg[k])

    for t in range(P.T):
        cl, cs = regs[t]; ctr = list(cl) + list(cs)
        for k, (d, i) in enumerate(order):
            tgt = x[P.q(t, d, i)]
            a = act[k]
            reach = {enc(L, S) for (L, S) in a}
            dec = {enc(L, S): v for (L, S), v in a.items()}
            if any(v != 0 for v in dec.values()):
                qc.ry(th, tgt)
                for val, ang in ((0, -th), (1, math.pi - th)):
                    targ = {s_ for s_, v in dec.items() if v == val}
                    if not targ: continue
                    for mask, value in _cover(targ, targ, reach, w):
                        cbits = [b for b in range(w) if (mask >> b) & 1]
                        if not cbits:
                            qc.ry(ang, tgt); continue
                        state = ''.join(str((value >> b) & 1) for b in reversed(cbits))
                        qc.append(RYGate(ang).control(len(cbits), ctrl_state=state), [ctr[b] for b in cbits] + [tgt])
            inc(cl if d == 0 else cs, tgt)          # counters end at the full counts of the period
    return qc


def prep_circuit_onehot(P, q, name='A'):
    """Same preparation A_q as prep_circuit, but the per-period counters (#long, #short) are one-hot registers:
    every forcing condition is then controlled by at most two ancillas (L == l, S == s) instead of a binary
    comparison, and increments are chains of controlled swaps."""
    order, act = P.plan()
    mL = max(L for a in act for (L, S) in a) + 1; mS = max(S for a in act for (L, S) in a) + 1
    x = QuantumRegister(P.nq, 'x'); oL = QuantumRegister(mL, 'oL'); oS = QuantumRegister(mS, 'oS')
    qc = QuantumCircuit(x, oL, oS, name=name)
    th = 2 * math.asin(math.sqrt(q))

    def shift(reg, ctrl, top):                    # one-hot value v -> v+1 (v < top), controlled on ctrl
        for j in range(top, 0, -1):
            qc.cswap(ctrl, reg[j], reg[j - 1])

    def unshift(reg, ctrl, top):
        for j in range(1, top + 1):
            qc.cswap(ctrl, reg[j], reg[j - 1])

    for t in range(P.T):
        qc.x(oL[0]); qc.x(oS[0])                  # counters start at (0, 0)
        ops = []
        for k, (d, i) in enumerate(order):
            tgt = x[P.q(t, d, i)]
            dec = act[k]
            if any(v != 0 for v in dec.values()):
                qc.ry(th, tgt)
                covered = set()
                for val, ang in ((0, -th), (1, math.pi - th)):
                    targ = {st for st, v in dec.items() if v == val}
                    # whole rows / columns with the same decision -> single control
                    for l in sorted({l for l, _ in targ}):
                        row = {st for st in dec if st[0] == l}
                        if row <= targ and not (row & covered):
                            qc.append(RYGate(ang).control(1), [oL[l], tgt]); covered |= row
                    for s_ in sorted({s_ for _, s_ in targ}):
                        col = {st for st in dec if st[1] == s_}
                        if col <= targ and not (col & covered):
                            qc.append(RYGate(ang).control(1), [oS[s_], tgt]); covered |= col
                    for st in sorted(targ - covered):
                        qc.append(RYGate(ang).control(2), [oL[st[0]], oS[st[1]], tgt]); covered.add(st)
            if k < len(order) - 1:
                nxt = act[k + 1]
                if d == 0:
                    top = max(L for L, _ in nxt); shift(oL, tgt, top); ops.append((oL, tgt, top))
                else:
                    top = max(S for _, S in nxt); shift(oS, tgt, top); ops.append((oS, tgt, top))
        for reg, ctrl, top in reversed(ops):
            unshift(reg, ctrl, top)
        qc.x(oL[0]); qc.x(oS[0])
    return qc


def cost_layer(qc, qubits, h, J, gamma):
    for k, v in enumerate(h):
        if v: qc.rz(-gamma * v, qubits[k])           # e^{-i g v x} = e^{-i g v (1-Z)/2} ~ RZ(-g v)  (up to phase)
    for (a, b), v in J.items():
        if v:                                         # x_a x_b = (1 - Z_a - Z_b + Z_a Z_b)/4
            qc.rzz(gamma * v / 2, qubits[a], qubits[b])
            qc.rz(-gamma * v / 2, qubits[a]); qc.rz(-gamma * v / 2, qubits[b])


def x_mixer(qc, qubits, beta):
    for q in qubits: qc.rx(2 * beta, q)


def xy_edges(P, ring=None):
    """Ring (default) or path edges inside each (period, direction) block, grouped into matchings (layers).
    A path also connects every configuration of a block with the same count (irreducible), with n-1 edges."""
    ring = getattr(P, 'xy_ring', True) if ring is None else ring
    n = P.na; layers = []
    if n < 2: return layers
    for t in range(P.T):
        for d in (0, 1):
            edges = [(P.q(t, d, i), P.q(t, d, (i + 1) % n)) for i in range(n if (n > 2 and ring) else n - 1)]
            for e in edges:                                   # greedy edge colouring
                for L in layers:
                    if all(e[0] not in f and e[1] not in f for f in L): L.append(e); break
                else: layers.append([e])
    return layers


def xy_mixer(qc, P, qubits, beta):
    """exp(-i beta (XX+YY)/2) on every ring edge of each (period, direction) block: conserves #long and
    #short of every period, hence feasibility."""
    for L in xy_edges(P):
        for a, b in L:
            qc.append(XXPlusYYGate(2 * beta), [qubits[a], qubits[b]])


def grover_mixer(qc, A, beta):
    """A e^{-i beta |0><0|} A^dag on all qubits of A (ancillas return to |0>)."""
    allq = list(qc.qubits[:A.num_qubits])
    qc.compose(A.inverse(), allq, inplace=True)
    xs = allq[:A.num_qubits]
    qc.x(xs)
    qc.mcp(-beta, xs[:-1], xs[-1])
    qc.x(xs)
    qc.compose(A, allq, inplace=True)


def qaoa_circuit(P, kind, params, q=0.5, h=None, J=None, nq_total=None, measure=False, counter='binary'):
    """counter: 'binary' (uncomputed, shared), 'onehot', or 'lean' (per-period, not uncomputed; XY / X / prep only)."""
    """kind in {'penalty_slack', 'penalty_unbal', 'warm_x', 'cg_grover', 'cg_xy'}; params = (gammas, betas)."""
    gam, bet = params
    if kind in ('penalty_slack', 'penalty_unbal'):
        n = nq_total or P.nq
        qc = QuantumCircuit(n, name=kind)
        qc.h(range(n))
        for g, b in zip(gam, bet):
            cost_layer(qc, list(range(n)), h, J, g); x_mixer(qc, list(range(n)), b)
    else:
        A = {'onehot': prep_circuit_onehot, 'lean': prep_circuit_lean}.get(counter, prep_circuit)(P, q)
        assert not (counter == 'lean' and kind == 'cg_grover'), 'lean preparation has no inverse'
        qc = QuantumCircuit(A.num_qubits, name=kind)
        qc.compose(A, inplace=True)
        xs = list(range(P.nq))
        for g, b in zip(gam, bet):
            cost_layer(qc, xs, h, J, g)
            if kind == 'warm_x': x_mixer(qc, xs, b)
            elif kind == 'cg_xy': xy_mixer(qc, P, xs, b)
            else: grover_mixer(qc, A, b)
    if measure:
        qc.measure_all()
    return qc
