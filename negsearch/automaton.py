"""General theory made executable: constraint automaton -> negation-forced state A_q -> amplitude amplification / QAOA.

Convention: a bit string is an integer with variable 0 as the MOST significant bit (index = sum_k x_k 2^(n-1-k)).
`to_qiskit` converts a length-2^n array to Qiskit's little-endian order (variable k = qubit k).

Core objects
  Problem        feasible set F (bool array), objective `cost` (minimised), `viol` (>=0, zero iff feasible).
  Automaton      minimal layered automaton of F for the fixed variable order (Myhill-Nerode classes of prefixes =
                 reduced OBDD).  State of a prefix = the set of feasible suffixes; dead state = empty set.
  prep_state     the negation-forced state  A_q|0> = sum_{x in F} sqrt(mu_q(x)) |x>  (support exactly F).
  compile_Aq     gate-level circuit of A_q from the automaton (controlled-RY on a binary state register).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------------------------------------
@dataclass
class Problem:
    name: str
    n: int
    feas: np.ndarray            # bool, length 2^n
    cost: np.ndarray            # float, minimised
    viol: np.ndarray            # float >= 0, zero iff feasible (generic penalty measure)
    meta: dict = field(default_factory=dict)

    @property
    def nF(self):
        return int(self.feas.sum())

    @property
    def opt(self):
        return float(self.cost[self.feas].min())

    @property
    def worst(self):
        return float(self.cost[self.feas].max())

    def good(self, rank=0.0):
        """Marked set: feasible and cost <= opt + rank*(worst-opt).  rank=0 -> optimal solutions."""
        thr = self.opt + rank * (self.worst - self.opt) + 1e-9
        return self.feas & (self.cost <= thr)


def bits_of(n):
    idx = np.arange(2 ** n)
    return ((idx[:, None] >> (n - 1 - np.arange(n))) & 1).astype(np.int64)


def bitrev_perm(n):
    """perm[i_msb] = i_lsb : reorders an MSB-first array into Qiskit order via arr_q[perm] = arr."""
    idx = np.arange(2 ** n)
    out = np.zeros_like(idx)
    for k in range(n):
        out |= ((idx >> (n - 1 - k)) & 1) << k
    return out


def to_qiskit(arr, n):
    out = np.empty_like(arr)
    out[bitrev_perm(n)] = arr
    return out


def from_qiskit(arr, n):
    return arr[bitrev_perm(n)]


# ----------------------------------------------------------------------------------------------
@dataclass
class Automaton:
    n: int
    state_of: list              # state_of[k][prefix] (prefix in [0,2^k)) -> class id (arbitrary), level 0..n
    dead: list                  # dead[k][class id] -> bool
    live: list                  # live[k][prefix] -> (bool b0 live, bool b1 live)  for k in 0..n-1
    # compact tables over live classes (renumbered 0..w_k-1)
    cls: list                   # cls[k]: array prefix -> compact id or -1 (dead)
    width: list                 # width[k] = number of live classes at level k
    trans: list                 # trans[k][s] = (t0, t1) compact ids at level k+1, -1 if dead

    @property
    def w(self):
        return max(self.width)


def build_automaton(feas: np.ndarray, n: int) -> Automaton:
    state_of, dead, cls, width = [], [], [], []
    for k in range(n + 1):
        rows = feas.reshape(2 ** k, 2 ** (n - k))
        uniq, inv = np.unique(rows, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        isdead = ~uniq.any(axis=1)
        state_of.append(inv); dead.append(isdead)
        new = -np.ones(len(uniq), int); c = 0
        for j in range(len(uniq)):
            if not isdead[j]:
                new[j] = c; c += 1
        cls.append(new[inv]); width.append(c)
    live = []
    trans = []
    for k in range(n):
        child = cls[k + 1]
        p = np.arange(2 ** k)
        l0 = child[2 * p] >= 0
        l1 = child[2 * p + 1] >= 0
        live.append((l0, l1))
        T = [[-1, -1] for _ in range(width[k])]
        seen = set()
        for pp in range(2 ** k):
            s = cls[k][pp]
            if s < 0 or s in seen:
                continue
            seen.add(s)
            T[s] = [int(child[2 * pp]), int(child[2 * pp + 1])]
        trans.append(T)
    return Automaton(n, state_of, dead, live, cls, width, trans)


def mu_q(A: Automaton, q: float = 0.5):
    """Distribution over x of the negation-forced walk with coin q (0 outside F) and decisions D(x)."""
    n = A.n
    idx = np.arange(2 ** n)
    mu = np.ones(2 ** n)
    D = np.zeros(2 ** n, int)
    for k in range(n):
        pref = idx >> (n - k)
        b = (idx >> (n - 1 - k)) & 1
        l0 = A.live[k][0][pref]; l1 = A.live[k][1][pref]
        both = l0 & l1
        f = np.where(both, np.where(b == 1, q, 1 - q), 1.0)
        ok = np.where(b == 1, l1, l0)
        mu *= np.where(ok, f, 0.0)
        D += both.astype(int)
    return mu, D


def prep_state(A: Automaton, q: float = 0.5):
    mu, _ = mu_q(A, q)
    return np.sqrt(mu)


# ----------------------------------------------------------------------------------------------
# generic numerics: amplitude amplification and QAOA
# ----------------------------------------------------------------------------------------------
def grover_curve(psi0, good, kmax):
    """Success probability after k = 0..kmax amplification iterations from psi0 (real, normalised)."""
    a = float((psi0[good] ** 2).sum())
    th = math.asin(math.sqrt(min(1.0, a)))
    return np.array([math.sin((2 * k + 1) * th) ** 2 for k in range(kmax + 1)]), a


def best_k(a):
    th = math.asin(math.sqrt(a))
    return max(0, int(math.floor(math.pi / (4 * th))))


def gm_qaoa_state(psi0, cost, params):
    """Grover-mixer QAOA on the support of psi0 (exact):  prod_l  e^{-i b_l |psi0><psi0|} e^{-i g_l C}."""
    p = len(params) // 2
    psi = psi0.astype(complex).copy()
    for l in range(p):
        psi = psi * np.exp(-1j * params[l] * cost)
        psi = psi - (1 - np.exp(-1j * params[p + l])) * psi0 * np.vdot(psi0, psi)
    return psi


def x_qaoa_state(cost, params, n):
    """Standard penalty QAOA: |+>^n, diag phase, X mixer."""
    p = len(params) // 2
    psi = np.ones(2 ** n, complex) / math.sqrt(2 ** n)
    for l in range(p):
        psi = psi * np.exp(-1j * params[l] * cost)
        c, s = math.cos(params[p + l]), -1j * math.sin(params[p + l])
        t = psi.reshape([2] * n)
        for ax in range(n):
            a0 = np.take(t, 0, axis=ax); a1 = np.take(t, 1, axis=ax)
            t = np.stack([c * a0 + s * a1, s * a0 + c * a1], axis=ax)
        psi = t.reshape(-1)
    return psi


def neighbour_hamiltonian(feas, n, dist2=True):
    """Adjacency of F under single flips and (optionally) swaps (Hamming distance <= 2 with equal weight), as a dense
    matrix on the feasible subspace.  A generic constraint-preserving mixer for ANY F (numerics only)."""
    idx = np.flatnonzero(feas)
    pos = -np.ones(2 ** n, int); pos[idx] = np.arange(len(idx))
    H = np.zeros((len(idx), len(idx)))
    for i, x in enumerate(idx):
        for k in range(n):
            y = x ^ (1 << (n - 1 - k))
            if feas[y]:
                H[i, pos[y]] = 1
        if dist2:
            for k in range(n):
                for m in range(k + 1, n):
                    bk = (x >> (n - 1 - k)) & 1; bm = (x >> (n - 1 - m)) & 1
                    if bk != bm:
                        y = x ^ (1 << (n - 1 - k)) ^ (1 << (n - 1 - m))
                        if feas[y]:
                            H[i, pos[y]] = 1
    return idx, H


def nb_qaoa_state(psi0_full, feas, cost, params, evals, evecs, idx, n):
    """Constraint-preserving mixer from the neighbour Hamiltonian, evolution on the feasible subspace."""
    p = len(params) // 2
    psi = psi0_full[idx].astype(complex)
    c = cost[idx]
    for l in range(p):
        psi = psi * np.exp(-1j * params[l] * c)
        psi = evecs @ (np.exp(-1j * params[p + l] * evals) * (evecs.T @ psi))
    out = np.zeros(2 ** n, complex); out[idx] = psi
    return out


# ----------------------------------------------------------------------------------------------
# gate-level compiler
# ----------------------------------------------------------------------------------------------
def _code(s, nb):
    return [(s >> j) & 1 for j in range(nb)]       # little-endian bits of the state id


def compile_Aq(A: Automaton, q: float = 0.5, uncompute: bool = False, interleave: bool = False, native: bool = False):
    """Qiskit circuit of A_q from the automaton.  Qubits: x_0..x_{n-1}, then a binary state register per level
    1..n-1 (levels of width 1 need none).  With interleave=True the registers sit next to the variables
    (x_0, r_1, x_1, r_2, ...), which keeps the MPS bond dimension of the state at most the automaton width.
    native=True emits each multi-controlled operation as one exact unitary gate (for MPS simulation, which would otherwise
    create large intermediate entanglement in the gate decomposition); native=False emits standard multi-controlled gates.
    Returns (circuit, info): info['x'] = x qubit indices, info['regs'] = {level: register qubit indices}."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RYGate, XGate, UnitaryGate
    from qiskit.quantum_info import Operator

    n = A.n
    theta = 2 * math.asin(math.sqrt(q))
    nbits = {k: (math.ceil(math.log2(A.width[k])) if A.width[k] > 1 else 0) for k in range(1, n)}
    total = n + sum(nbits.values())
    qc = QuantumCircuit(total)

    def mc(base, nctrl, state, qubits):
        g = base.control(nctrl, ctrl_state=state)
        qc.append(UnitaryGate(Operator(g).data) if native else g, qubits)

    xq, regs = [], {}
    if interleave:
        pos = 0
        for k in range(n):
            if nbits.get(k, 0):
                regs[k] = list(range(pos, pos + nbits[k])); pos += nbits[k]
            xq.append(pos); pos += 1
    else:
        xq = list(range(n)); pos = n
        for k in range(1, n):
            if nbits[k]:
                regs[k] = list(range(pos, pos + nbits[k])); pos += nbits[k]

    def ctrl_pattern(k, s):
        """qubits and ctrl_state integer selecting 'register at level k == s'"""
        if k not in regs:
            return [], 0
        nb = len(regs[k])
        return list(regs[k]), sum(b << j for j, b in enumerate(_code(s, nb)))

    def update(k, s, b, t, target_level):
        """flip the bits of the level-(k+1) register that are 1 in the code of t, controlled on (r_k == s, x_k == b)"""
        nb = len(regs[target_level])
        cq, cs = ctrl_pattern(k, s)
        ctrls = cq + [xq[k]]
        state = cs | (b << len(cq))
        for j, bit in enumerate(_code(t, nb)):
            if bit:
                mc(XGate(), len(ctrls), state, ctrls + [regs[target_level][j]])

    for k in range(n):
        for s in range(A.width[k]):
            t0, t1 = A.trans[k][s]
            cq, cs = ctrl_pattern(k, s)
            if t0 >= 0 and t1 >= 0:
                if cq:
                    mc(RYGate(theta), len(cq), cs, cq + [xq[k]])
                else:
                    qc.ry(theta, xq[k])
            elif t1 >= 0 and t0 < 0:
                if cq:
                    mc(XGate(), len(cq), cs, cq + [xq[k]])
                else:
                    qc.x(xq[k])
        if k + 1 in regs:
            for s in range(A.width[k]):
                for b in (0, 1):
                    t = A.trans[k][s][b]
                    if t >= 0:
                        update(k, s, b, t, k + 1)
    if uncompute:           # run the register updates backwards: ancillas return to |0>, state = sum sqrt(mu)|x>|0>
        for k in range(n - 1, 0, -1):
            if k not in regs:
                continue
            for s in range(A.width[k - 1]):
                for b in (0, 1):
                    t = A.trans[k - 1][s][b]
                    if t >= 0:
                        update(k - 1, s, b, t, k)
    return qc, dict(x=list(xq), regs=regs, nq=qc.num_qubits)


def reflect_zero(qc, qubits, beta):
    """e^{-i beta |0..0><0..0|} on the given qubits (global phase aside)."""
    for q in qubits:
        qc.x(q)
    if len(qubits) == 1:
        qc.p(-beta, qubits[0])
    else:
        from qiskit.circuit.library import PhaseGate
        qc.append(PhaseGate(-beta).control(len(qubits) - 1, ctrl_state=None), list(qubits))
    for q in qubits:
        qc.x(q)


def append_diag(qc, qubits, phases_qiskit_order):
    from qiskit.circuit.library import Diagonal
    qc.append(Diagonal(phases_qiskit_order), qubits)


def gm_mixer(qc, Aq, beta):
    """A (e^{-i beta |0><0|}) A^dagger over all qubits of the circuit."""
    qc.compose(Aq.inverse(), inplace=True)
    reflect_zero(qc, list(range(qc.num_qubits)), beta)
    qc.compose(Aq, inplace=True)


def marginal_x(sv, n, nq):
    """Probability over x (MSB-first order) from a Qiskit statevector of nq qubits; also returns the
    max probability mass outside the dominant ancilla pattern (garbage determinism check)."""
    p = np.abs(np.asarray(sv.data if hasattr(sv, 'data') else sv)) ** 2
    p = p.reshape(2 ** (nq - n), 2 ** n)       # rows: ancilla bits (high), cols: x (low qubits)
    px = p.sum(axis=0)
    return from_qiskit(px, n)


def compile_feas_oracle(A: Automaton):
    """Phase oracle that flips the sign of every FEASIBLE string x (what standard Grover needs besides the objective test):
    run the automaton into fresh registers (with a dead code per level), phase on 'accepted', run backwards.
    Qubits: x_0..x_{n-1}, then one register per level 1..n."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import XGate

    n = A.n
    nbits = {k: max(1, math.ceil(math.log2(A.width[k] + 1))) for k in range(1, n + 1)}
    total = n + sum(nbits.values())
    fwd = QuantumCircuit(total)
    regs, pos = {}, n
    for k in range(1, n + 1):
        regs[k] = list(range(pos, pos + nbits[k])); pos += nbits[k]
    dead = {k: A.width[k] for k in range(n + 1)}

    def sel(k, s):
        if k == 0:
            return [], 0
        nb = len(regs[k])
        return list(regs[k]), sum(b << j for j, b in enumerate(_code(s, nb)))

    for k in range(n):
        states = list(range(A.width[k])) + ([dead[k]] if k > 0 else [])
        for s in states:
            for b in (0, 1):
                if s == dead[k]:
                    if b == 1:
                        continue                      # the dead state ignores x_k: one update covers both values
                    t = dead[k + 1]
                    cq, cs = sel(k, s); ctrls, state = cq, cs
                else:
                    t = A.trans[k][s][b]
                    t = dead[k + 1] if t < 0 else t
                    cq, cs = sel(k, s); ctrls, state = cq + [k], cs | (b << len(cq))
                for j, bit in enumerate(_code(t, len(regs[k + 1]))):
                    if bit:
                        g = XGate().control(len(ctrls), ctrl_state=state)
                        fwd.append(g, ctrls + [regs[k + 1][j]])
    qc = fwd.copy()
    flag = regs[n][0]                                  # level-n code 0 = accepted, 1 = dead
    qc.x(flag); qc.z(flag); qc.x(flag)                 # sign -1 on accepted
    qc.compose(fwd.inverse(), inplace=True)
    return qc, dict(x=list(range(n)), regs=regs)


def mcz_vchain(qc, qubits, anc):
    """phase flip on |1...1> of `qubits` with a Toffoli V-chain (needs len(qubits)-3 clean ancillas)."""
    qubits = list(qubits)
    if len(qubits) == 1:
        qc.z(qubits[0]); return
    if len(qubits) == 2:
        qc.cz(*qubits); return
    last = qubits[-1]
    qc.h(last)
    qc.mcx(qubits[:-1], last, list(anc)[:max(0, len(qubits) - 3)], mode='v-chain')
    qc.h(last)


def grover_iteration_circuit(A, good, q=0.5, vchain=True):
    """One amplification iteration on A_q (marks `good`, an array over x in MSB-first order): returns (circuit, n_state_qubits).
    The circuit is  A_q  S_good  A_q (2|0><0|-I) A_q^dagger ... i.e. prepared state followed by one iteration, measured on x."""
    import math
    from qiskit import QuantumCircuit
    n = A.n
    Aq, info = compile_Aq(A, q)
    N = Aq.num_qubits
    nanc = max(0, N - 3) if vchain else 0
    qc = QuantumCircuit(N + nanc)
    qc.compose(Aq, qubits=range(N), inplace=True)
    append_diag(qc, list(range(n)), to_qiskit(np.where(good, -1.0, 1.0).astype(complex), n))
    qc.compose(Aq.inverse(), qubits=range(N), inplace=True)
    for k in range(N):
        qc.x(k)
    if vchain:
        mcz_vchain(qc, range(N), range(N, N + nanc))
    else:
        reflect_zero(qc, list(range(N)), math.pi)
    for k in range(N):
        qc.x(k)
    qc.compose(Aq, qubits=range(N), inplace=True)
    return qc, N
