"""Hardware circuits for the multi-constraint portfolio case with classical conditioning on the interface.
Interface = per (period, sector) block counts (l, s).  Given an interface tuple, the state is a PRODUCT of small block states; each block is
prepared by the generic automaton compiler (binary or one-hot register, whichever has fewer CZ) and the QAOA layer is
   phase(cost on all x qubits) -> block Grover mixer (A_b^dagger, reflection about |0>, A_b) for every block,
which conserves every block's constraints, so every sample is feasible by construction (Theorem on feasibility invariance, case (b))."""
from __future__ import annotations

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit

from .automaton import compile_Aq
from .depth_study import compile_Aq_onehot, n_ancillas, mcz_vchain, reflect_zero
from .large_case import sector_block_automaton, count_cz
from .structural import count_feasible


def block_choice(size, l, s):
    """compile the block with both register encodings, keep the one with fewer CZ.  Returns (circuit, automaton, encoding)"""
    B = sector_block_automaton(size, l, s)
    cb = compile_Aq(B, 0.5)[0]; co = compile_Aq_onehot(B)[0]
    return (cb, B, 'binary') if count_cz(cb)[2] <= count_cz(co)[2] else (co, B, 'one-hot')


def block_strings(B):
    """accepted strings of a block automaton with their coin-walk probability mu(x) (decision = both children alive)"""
    out = []

    def rec(k, s, x, mu):
        if k == B.n:
            out.append((x, mu)); return
        t0, t1 = B.trans[k][s]
        if t0 >= 0 and t1 >= 0:
            rec(k + 1, t0, x + (0,), mu * 0.5); rec(k + 1, t1, x + (1,), mu * 0.5)
        elif t0 >= 0:
            rec(k + 1, t0, x + (0,), mu)
        else:
            rec(k + 1, t1, x + (1,), mu)
    rec(0, 0, (), 1.0)
    return out


def interface_tuples(A, T, sectors, Dmax=3, Bmax=3, Q=None, cap=1):
    """all feasible interface tuples ((l,s) per period and sector) with weight |group|/|F|"""
    Q = Q if Q is not None else int(round(0.6 * T * Bmax))
    combos = {}
    for size in set(sectors):
        combos[size] = []
        for l in range(cap + 1):
            for s in range(Bmax + 1):
                B = sector_block_automaton(size, l, s)
                if B.width[B.n] > 0 and count_feasible(B) > 0:
                    combos[size].append((l, s, count_feasible(B)))
    per_period = []
    for choice in itertools.product(*[combos[sz] for sz in sectors]):
        L = sum(c[0] for c in choice); S = sum(c[1] for c in choice)
        if 0 <= L - S <= Dmax and L + S <= Bmax:
            w = 1
            for c in choice:
                w *= c[2]
            per_period.append((tuple((c[0], c[1]) for c in choice), L + S, w))
    out = []
    for combo in itertools.product(per_period, repeat=T):
        if sum(c[1] for c in combo) <= Q:
            w = 1
            for c in combo:
                w *= c[2]
            out.append((tuple(c[0] for c in combo), w))
    tot = sum(w for _, w in out)
    return [(k, w / tot) for k, w in out], tot


def x_index(A, t, i, d):
    return (t * A + i) * 2 + d


def qubo_cost(A, T, seed):
    """cost as QUBO over the x variables: returns constant, h (linear), J (dict (i,j)->coef) for c(x)=const+h.x+sum J x_i x_j  (same model as the numerics)"""
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.02, 0.12, A); Gm = rng.normal(size=(A, A)); Sg = Gm @ Gm.T / A * 0.05
    nv = 2 * A * T
    h = np.zeros(nv); J = {}

    def addq(u, v, w):
        """w * (u.x)(v.x) for sparse linear forms u, v given as {var: coef}"""
        for a, ca in u.items():
            for b, cb in v.items():
                if a == b:
                    h[a] += w * ca * cb          # x^2 = x
                else:
                    key = (min(a, b), max(a, b)); J[key] = J.get(key, 0.0) + w * ca * cb
    y = lambda t, i: {x_index(A, t, i, 0): 1.0, x_index(A, t, i, 1): -1.0}
    for t in range(T):
        for i in range(A):
            for j in range(A):
                addq(y(t, i), y(t, j), 8 * Sg[i, j])
            for v, cf in y(t, i).items():
                h[v] += -10 * mu[i] * cf
    for t in range(1, T):
        for i in range(A):
            d = {}
            for v, cf in y(t, i).items():
                d[v] = d.get(v, 0) + cf
            for v, cf in y(t - 1, i).items():
                d[v] = d.get(v, 0) - cf
            addq(d, d, 0.08)
    return h, J


def ising_from_qubo(h, J):
    """x=(1-z)/2: returns {(i,):c, (i,j):c} (qubit k = variable k) and the constant offset (dropped: global phase)"""
    co = {}
    for i, v in enumerate(h):
        if v:
            co[(i,)] = co.get((i,), 0) - v / 2
    for (i, j), v in J.items():
        co[(i,)] = co.get((i,), 0) - v / 4; co[(j,)] = co.get((j,), 0) - v / 4
        co[(i, j)] = co.get((i, j), 0) + v / 4
    return {k: c for k, c in co.items() if abs(c) > 1e-12}


def build_hw_circuit(A, T, sectors, iface, coeffs, params, measure=True):
    """full circuit for one interface tuple.  Qubits: x (2*A*T), then block registers, then one shared ancilla pool (V-chain)"""
    from .depth_study import phase_poly
    nx = 2 * A * T
    blocks = []                                   # (x qubits, circuit, enc)
    pos = nx
    sec_off = np.cumsum([0] + list(sectors))
    for t in range(T):
        for si, size in enumerate(sectors):
            l, s = iface[t][si]
            circ, B, enc = block_choice(size, l, s)
            xq = [x_index(A, t, sec_off[si] + j, d) for j in range(size) for d in (0, 1)]
            extra = list(range(pos, pos + circ.num_qubits - 2 * size)); pos += len(extra)
            blocks.append((xq + extra, circ, enc))
    pool = max(n_ancillas(c.num_qubits) for _, c, _ in blocks)
    qc = QuantumCircuit(pos + pool)
    anc = list(range(pos, pos + pool))
    for qs, c, _ in blocks:
        qc.compose(c, qubits=qs, inplace=True)
    p = len(params) // 2
    for l in range(p):
        phase_poly(qc, coeffs, params[l])
        for qs, c, _ in blocks:
            qc.compose(c.inverse(), qubits=qs, inplace=True)
            reflect_zero(qc, len(qs), anc[:n_ancillas(len(qs))], params[p + l]) if False else _reflect_on(qc, qs, anc, params[p + l])
            qc.compose(c, qubits=qs, inplace=True)
    if measure:
        from qiskit.circuit import ClassicalRegister
        cr = ClassicalRegister(nx, 'x'); qc.add_register(cr)
        for k in range(nx):
            qc.measure(k, cr[k])
    return qc, blocks


def _reflect_on(qc, qs, anc, beta):
    """e^{-i beta |0..0><0..0|} on the qubits qs with V-chain ancillas taken from the shared pool"""
    N = len(qs)
    need = n_ancillas(N)
    a = anc[:need]
    for q in qs:
        qc.x(q)
    if abs(beta - math.pi) < 1e-12:
        mcz_vchain(qc, qs, a)
    else:
        controls = list(qs[:-1]); tgt = a[-1] if a else None
        chain = a[:-1]
        if N == 1:
            qc.p(-beta, qs[0])
        else:
            qc.mcx(controls, tgt, chain[:max(0, len(controls) - 2)], mode='v-chain') if len(controls) > 2 else qc.mcx(controls, tgt)
            qc.cp(-beta, tgt, qs[-1])
            qc.mcx(controls, tgt, chain[:max(0, len(controls) - 2)], mode='v-chain') if len(controls) > 2 else qc.mcx(controls, tgt)
    for q in qs:
        qc.x(q)


# ---------------------------------------------------------------------------------------------------------------------
# Cheap feasibility-preserving block mixer: exchange of the (long, short) states of two assets of the same sector.
# P = SWAP(long_j, long_k) * SWAP(short_j, short_k) is an involution that maps feasible strings of the block to feasible strings
# (it permutes the assets, so the counts (l, s) and the exclusivity are conserved), hence e^{-i beta P} = cos(beta) I - i sin(beta) P
# keeps every sample feasible; every arrangement of the block is reachable by such exchanges.  Needs no copy of A_q and no register.
def pair_swap_mixer(qc, lj, sj, lk, sk, beta):
    """exp(-i beta P) on the four qubits (exact).  P = S1 S2, S_i = 1 - 2|singlet_i><singlet_i|.
    W maps the singlet of a pair to |11> (CX then H on the first qubit), so S_i = W^dag CZ W and exp(-i beta S1 S2) is
    W^dag exp(-i beta (-1)^{a1 b1 + a2 b2}) W; with g = a1b1 xor a2b2 = a1b1 + a2b2 - 2 a1b1a2b2 the diagonal phase is
    e^{-i beta} e^{2 i beta g}, built from two controlled phases and one four-qubit controlled phase."""
    for (p, q) in ((lj, lk), (sj, sk)):
        qc.cx(p, q); qc.h(p)
    # after W: (a1, b1) = (lj, lk) wires, (a2, b2) = (sj, sk) wires
    qc.cp(2 * beta, lj, lk)
    qc.cp(2 * beta, sj, sk)
    qc.mcp(-4 * beta, [lj, lk, sj], sk)
    for (p, q) in ((lj, lk), (sj, sk)):
        qc.h(p); qc.cx(p, q)


def swap_pairs(m, pairs='all'):
    """asset pairs exchanged in a block of m assets: all pairs, or a path (neighbours only; still connects every arrangement)"""
    return [(j, k) for j in range(m) for k in range(j + 1, m)] if pairs == 'all' else [(j, j + 1) for j in range(m - 1)]


def swap_mixer_layer(qc, xq, beta, pairs='all'):
    """pair exchanges inside one block; xq = [long_0, short_0, long_1, short_1, ...]"""
    m = len(xq) // 2
    for j, k in swap_pairs(m, pairs):
        pair_swap_mixer(qc, xq[2 * j], xq[2 * j + 1], xq[2 * k], xq[2 * k + 1], beta)


def block_choice_uncompute(size, l, s):
    B = sector_block_automaton(size, l, s)
    return compile_Aq(B, 0.5, uncompute=True)[0], B


def build_hw_circuit_swap(A, T, sectors, iface, coeffs, params, measure=True, pairs='all'):
    """preparation (registers uncomputed) + p layers of [objective phase, block swap mixers]"""
    from .depth_study import phase_poly
    nx = 2 * A * T
    blocks = []; pos = nx
    sec_off = np.cumsum([0] + list(sectors))
    for t in range(T):
        for si, size in enumerate(sectors):
            l, s = iface[t][si]
            circ, B = block_choice_uncompute(size, l, s)
            xq = [x_index(A, t, sec_off[si] + j, d) for j in range(size) for d in (0, 1)]
            extra = list(range(pos, pos + circ.num_qubits - 2 * size)); pos += len(extra)
            blocks.append((xq + extra, xq, circ))
    qc = QuantumCircuit(pos)
    for qs, xq, c in blocks:
        qc.compose(c, qubits=qs, inplace=True)
    p = len(params) // 2
    for l in range(p):
        phase_poly(qc, coeffs, params[l])
        for qs, xq, c in blocks:
            swap_mixer_layer(qc, xq, params[p + l], pairs)
    if measure:
        from qiskit.circuit import ClassicalRegister
        cr = ClassicalRegister(nx, 'x'); qc.add_register(cr)
        for k in range(nx):
            qc.measure(k, cr[k])
    return qc, blocks


def tensor_qaoa_swap(blocks, combos, c_phase, params, pairs='all'):
    """exact QAOA on the product space of the blocks: phase on c_phase, then sequential pair-exchange mixers inside every block
    (same order as the circuit).  blocks = [(t, sector, [(string, mu), ...]), ...]; returns probabilities over the product space"""
    p = len(params) // 2
    shape = [len(b[2]) for b in blocks]
    amps = [np.sqrt(np.array([m for _, m in b[2]])) for b in blocks]
    psi = amps[0]
    for a in amps[1:]:
        psi = np.multiply.outer(psi, a)
    psi = psi.astype(complex); cT = np.zeros(shape); cT[tuple(combos.T)] = c_phase
    perms = []
    for (t, si, strs) in blocks:
        ix = {s: i for i, (s, _) in enumerate(strs)}; m = len(strs[0][0]) // 2; pl = []
        for j, k in swap_pairs(m, pairs):
            perm = []
            for s, _ in strs:
                s2 = list(s); s2[2 * j:2 * j + 2], s2[2 * k:2 * k + 2] = s[2 * k:2 * k + 2], s[2 * j:2 * j + 2]; perm.append(ix[tuple(s2)])
            perm = np.array(perm); inv = np.empty_like(perm); inv[perm] = np.arange(len(perm)); pl.append(inv)
        perms.append(pl)
    for l in range(p):
        psi = psi * np.exp(-1j * params[l] * cT)
        for ax, pl in enumerate(perms):
            for inv in pl:
                psi = np.cos(params[p + l]) * psi - 1j * np.sin(params[p + l]) * np.take(psi, inv, axis=ax)
    return np.abs(psi) ** 2


def prune(co, frac):
    """drop two-body Ising couplings with |coefficient| < frac * max |coefficient| (feasibility is unaffected: only the objective changes)"""
    m = max(abs(c) for k, c in co.items() if len(k) == 2)
    return {k: c for k, c in co.items() if len(k) == 1 or abs(c) >= frac * m}
