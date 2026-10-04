"""
dynsearch: dynamic-circuit-native amplitude amplification for general constraint problems.

Two ideas, both exact (they keep Grover's quadratic speedup):

A) Measurement-based uncomputation (MBU) of every AND/XOR ancilla in the oracle and the
   diffuser. An ancilla holding f(x) is measured in the X basis; if the outcome is 1 the
   phase (-1)^f(x) is fixed with a classically controlled CZ/Z (Gidney 2018 trick,
   generalised here to arbitrary CNF + XOR constraints and to the diffuser).
   Logic is identical to coherent uncomputation; only the cost changes.

B) Heralded weakly-measured search: inside each oracle call the satisfaction flag f(x) is
   weakly copied to a herald qubit (CRy(2*phi_t)) and measured mid-circuit. A click projects
   the search register onto the solution subspace (the solution is certified *before* reading
   x). The schedule sin^2(phi_t) = c/(t+1) needs no knowledge of the number of solutions M.
   All later iterations are skipped with classical feed-forward.

Problems are given as constraints over n Boolean variables:
    clauses : tuples of signed 1-based ints (DIMACS style), e.g. (1, -3, 4)  -> x1 or not x3 or x4
    xors    : (vars_tuple_1based, rhs)  ->  XOR of vars == rhs
"""
from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RCCXGate
from qiskit.circuit.classical import expr


# ----------------------------------------------------------------------------------------
# Problem description
# ----------------------------------------------------------------------------------------
@dataclass
class Problem:
    n: int
    clauses: list = field(default_factory=list)
    xors: list = field(default_factory=list)
    name: str = "problem"

    @property
    def m(self) -> int:
        return len(self.clauses) + len(self.xors)

    def is_solution(self, bits) -> bool:
        """bits[i] is the value of variable i+1."""
        for cl in self.clauses:
            if not any((bits[abs(l) - 1] == 1) == (l > 0) for l in cl):
                return False
        for vs, rhs in self.xors:
            if sum(bits[v - 1] for v in vs) % 2 != rhs:
                return False
        return True

    def solutions(self):
        return [b for b in itertools.product((0, 1), repeat=self.n) if self.is_solution(b)]

    def check_bitstring(self, s: str) -> bool:
        """Qiskit bitstring of the 'out' register (rightmost char = variable 1)."""
        return self.is_solution([int(c) for c in s[::-1]])

    @staticmethod
    def from_dimacs(text: str, name="dimacs") -> "Problem":
        n, clauses = 0, []
        for line in text.splitlines():
            line = line.strip()
            if not line or line[0] in "c%":
                continue
            if line[0] == "p":
                n = int(line.split()[2])
                continue
            lits = [int(t) for t in line.split() if t != "0"]
            if lits:
                clauses.append(tuple(lits))
        return Problem(n, clauses, name=name)


def graph_2coloring(n_vertices, edges, encoding="xor"):
    if encoding == "xor":
        return Problem(n_vertices, xors=[((i + 1, j + 1), 1) for i, j in edges],
                       name=f"2col-xor(n={n_vertices},e={len(edges)})")
    cls = []
    for i, j in edges:
        cls += [(i + 1, j + 1), (-(i + 1), -(j + 1))]
    return Problem(n_vertices, clauses=cls, name=f"2col-cnf(n={n_vertices},e={len(edges)})")


def binary_sudoku_2x2(encoding="xor"):
    """The 2x2 binary Sudoku of the Qiskit textbook / original notebook (= 2-colouring of C4)."""
    p = graph_2coloring(4, [(0, 1), (0, 2), (1, 3), (2, 3)], encoding)
    p.name = f"sudoku2x2-{encoding}"
    return p


def graph_3coloring(n_vertices, edges):
    """2 bits per vertex (colour 0,1,2; pattern 11 forbidden). Produces 2- and 4-literal clauses."""
    var = lambda v, b: 2 * v + b + 1
    cls = [(-var(v, 0), -var(v, 1)) for v in range(n_vertices)]
    for v, w in edges:
        for c in range(3):
            b0, b1 = c & 1, (c >> 1) & 1
            lit = lambda u, b, val: -var(u, b) if val else var(u, b)
            cls.append((lit(v, 0, b0), lit(v, 1, b1), lit(w, 0, b0), lit(w, 1, b1)))
    return Problem(2 * n_vertices, clauses=cls, name=f"3col(n={n_vertices},e={len(edges)})")


def planted_ksat(n, m, k=3, seed=0):
    rng = random.Random(seed)
    hidden = [rng.randint(0, 1) for _ in range(n)]
    cls = []
    while len(cls) < m:
        vs = rng.sample(range(n), k)
        lits = tuple((v + 1) * (1 if rng.random() < 0.5 else -1) for v in vs)
        if any((hidden[abs(l) - 1] == 1) == (l > 0) for l in lits):
            cls.append(lits)
    return Problem(n, clauses=cls, name=f"planted-{k}SAT(n={n},m={m},seed={seed})")


# ----------------------------------------------------------------------------------------
# Circuit builder
# ----------------------------------------------------------------------------------------
class _Pool:
    def __init__(self, qubits):
        self.free = list(qubits)
        self.peak = 0
        self.size = len(qubits)

    def get(self):
        q = self.free.pop()
        self.peak = max(self.peak, self.size - len(self.free))
        return q

    def put(self, q):
        self.free.append(q)


class SearchBuilder:
    """
    mode = "mbu"      : measurement-based uncomputation (dynamic circuit)  [idea A]
    mode = "coherent" : same structure, coherent uncomputation (static baseline)
    """

    def __init__(self, problem: Problem, mode: str = "mbu", herald: int = 0,
                 and_gate: str = "rccx"):
        assert mode in ("mbu", "coherent") and and_gate in ("rccx", "ccx")
        self.p, self.mode, self.herald, self.and_gate = problem, mode, herald, and_gate
        n, m = problem.n, problem.m
        kmax = max([len(c) for c in problem.clauses] + [1])
        pool_size = max(kmax - 2, m - 2, n - 2, 1)
        self.x = QuantumRegister(n, "x")
        self.u = QuantumRegister(m, "u")          # one 'unsatisfied' bit per constraint
        self.w = QuantumRegister(pool_size, "w")  # work ancillas (AND chains)
        self.f = QuantumRegister(1, "f")          # global satisfaction flag
        regs = [self.x, self.u, self.w, self.f]
        if herald:
            self.a = QuantumRegister(1, "a")
            regs.append(self.a)
        self.mb = ClassicalRegister(1, "mb")      # scratch bit for MBU outcomes
        self.out = ClassicalRegister(n, "out")
        cregs = [self.mb, self.out]
        if herald:
            self.h = ClassicalRegister(int(herald), "h")   # one herald bit per iteration
            cregs.append(self.h)
        self.qc = QuantumCircuit(*regs, *cregs)
        self.pool = _Pool(self.w)

    # ---------- AND chain primitives -------------------------------------------------
    def _frame(self, ctrls, pols):
        neg = [q for q, p in zip(ctrls, pols) if not p]
        if neg:
            self.qc.x(neg)

    def _and2(self, a, b, t):
        """Exact logical AND into a CLEAN target: t = a*b.
        rccx leaves a phase i^(a*b) on a clean target; one S-dagger removes it (3 CX instead of 6)."""
        if self.and_gate == "rccx":
            self.qc.rccx(a, b, t)
            self.qc.sdg(t)
        else:
            self.qc.ccx(a, b, t)

    def _and2_inv(self, a, b, t):
        if self.and_gate == "rccx":
            self.qc.s(t)
            self.qc.append(RCCXGate().inverse(), [a, b, t])
        else:
            self.qc.ccx(a, b, t)

    def _chain(self, ctrls):
        """Compute t_1..t_{k-2} with t_i = AND(t_{i-1}, c_i), t_0 = c_0. Returns temps."""
        temps, prev = [], ctrls[0]
        for i in range(1, len(ctrls) - 1):
            t = self.pool.get()
            self._and2(prev, ctrls[i], t)
            temps.append(t)
            prev = t
        return temps

    def _clean_chain(self, ctrls, temps):
        qc = self.qc
        for i in range(len(temps), 0, -1):          # temps[i-1] = AND(prev, ctrls[i])
            t = temps[i - 1]
            prev = ctrls[0] if i == 1 else temps[i - 2]
            if self.mode == "coherent":
                self._and2_inv(prev, ctrls[i], t)
            else:
                qc.h(t)
                qc.measure(t, self.mb[0])
                with qc.if_test((self.mb[0], 1)):
                    qc.cz(prev, ctrls[i])
                qc.reset(t)
            self.pool.put(t)

    def and_compute(self, ctrls, pols, target):
        """target (clean) <- AND_i (ctrl_i == pol_i). Intermediates are cleaned immediately."""
        qc = self.qc
        self._frame(ctrls, pols)
        if len(ctrls) == 1:
            qc.cx(ctrls[0], target)
        else:
            temps = self._chain(ctrls)
            last = ctrls[0] if not temps else temps[-1]
            self._and2(last, ctrls[-1], target)
            self._clean_chain(ctrls, temps)
        self._frame(ctrls, pols)

    def and_uncompute(self, ctrls, pols, target):
        qc = self.qc
        self._frame(ctrls, pols)
        if len(ctrls) == 1:
            if self.mode == "coherent":
                qc.cx(ctrls[0], target)
            else:
                qc.h(target)
                qc.measure(target, self.mb[0])
                with qc.if_test((self.mb[0], 1)):
                    qc.z(ctrls[0])
        else:
            temps = self._chain(ctrls)
            last = ctrls[0] if not temps else temps[-1]
            if self.mode == "coherent":
                self._and2_inv(last, ctrls[-1], target)
            else:
                qc.h(target)
                qc.measure(target, self.mb[0])
                with qc.if_test((self.mb[0], 1)):
                    qc.cz(last, ctrls[-1])
            self._clean_chain(ctrls, temps)
        if self.mode == "mbu":
            qc.reset(target)
        self._frame(ctrls, pols)

    # ---------- constraints -----------------------------------------------------------
    def _constraints(self):
        x = self.x
        out = []
        for cl in self.p.clauses:  # unsat <=> every literal false
            out.append(("and", [x[abs(l) - 1] for l in cl], [l < 0 for l in cl]))
        for vs, rhs in self.p.xors:  # unsat <=> parity != rhs
            out.append(("xor", [x[v - 1] for v in vs], rhs))
        return out

    def _compute_u(self):
        for j, c in enumerate(self._constraints()):
            if c[0] == "and":
                self.and_compute(c[1], c[2], self.u[j])
            else:
                for q in c[1]:
                    self.qc.cx(q, self.u[j])
                if c[2] == 1:
                    self.qc.x(self.u[j])

    def _uncompute_u(self):
        qc = self.qc
        for j, c in reversed(list(enumerate(self._constraints()))):
            if c[0] == "and":
                self.and_uncompute(c[1], c[2], self.u[j])
            elif self.mode == "coherent":
                if c[2] == 1:
                    qc.x(self.u[j])
                for q in c[1]:
                    qc.cx(q, self.u[j])
            else:  # MBU of a parity: correction is a product of single-qubit Z's
                qc.h(self.u[j])
                qc.measure(self.u[j], self.mb[0])
                with qc.if_test((self.mb[0], 1)):
                    qc.z(c[1])
                qc.reset(self.u[j])

    # ---------- oracle / diffuser -----------------------------------------------------
    def oracle(self, herald_angle=None, herald_bit=None):
        qc = self.qc
        self._compute_u()
        uq = list(self.u)
        self.and_compute(uq, [False] * len(uq), self.f[0])
        qc.z(self.f[0])                                   # phase kick on solutions
        if herald_angle is not None:
            qc.cry(herald_angle, self.f[0], self.a[0])    # weak copy of the flag
        self.and_uncompute(uq, [False] * len(uq), self.f[0])
        self._uncompute_u()
        if herald_angle is not None:
            qc.measure(self.a[0], herald_bit)             # herald (before the diffuser!)
            qc.reset(self.a[0])

    def diffuser(self):
        qc, x, n = self.qc, list(self.x), self.p.n
        qc.h(x)
        if n == 1:
            qc.z(x[0])
        else:
            t = self.pool.get()
            self.and_compute(x[:-1], [False] * (n - 1), t)
            qc.x(x[-1]); qc.cz(t, x[-1]); qc.x(x[-1])
            self.and_uncompute(x[:-1], [False] * (n - 1), t)
            self.pool.put(t)
        qc.h(x)

    # ---------- algorithms ----------------------------------------------------------
    def grover(self, iterations: int) -> QuantumCircuit:
        self.qc.h(self.x)
        for _ in range(iterations):
            self.oracle()
            self.diffuser()
        self.qc.measure(self.x, self.out)
        return self.qc

    def heralded(self, T: int, c: float = 5.0) -> QuantumCircuit:
        """Weakly-measured search, schedule sin^2(phi_t) = min(1/2, c/(t+1)); no knowledge of M.
        Iteration t runs only if no earlier herald clicked (classical feed-forward)."""
        assert self.herald == T
        qc = self.qc
        qc.h(self.x)
        for t in range(T):
            phi = math.asin(math.sqrt(min(0.5, c / (t + 1))))
            if t == 0:
                self.oracle(herald_angle=2 * phi, herald_bit=self.h[0])
            else:
                with qc.if_test(self._none_clicked(t)):
                    self.oracle(herald_angle=2 * phi, herald_bit=self.h[t])
            with qc.if_test(self._none_clicked(t + 1)):
                self.diffuser()
        qc.measure(self.x, self.out)
        return qc

    def _none_clicked(self, t):
        cond = expr.lift(self.h[0])
        for i in range(1, t):
            cond = expr.logic_or(cond, self.h[i])
        return expr.logic_not(cond)


def grover_circuit(problem, iterations, mode="mbu", and_gate="rccx"):
    return SearchBuilder(problem, mode, and_gate=and_gate).grover(iterations)


def heralded_circuit(problem, T, c=5.0, mode="mbu", and_gate="rccx"):
    return SearchBuilder(problem, mode, herald=T, and_gate=and_gate).heralded(T, c)


def textbook_grover(problem, iterations):
    """Baseline in the style of the original notebook: qiskit mcx without work ancillas."""
    n, cons = problem.n, None
    x = QuantumRegister(n, "x"); u = QuantumRegister(problem.m, "u"); o = QuantumRegister(1, "o")
    out = ClassicalRegister(n, "out")
    qc = QuantumCircuit(x, u, o, out)
    qc.h(x); qc.x(o); qc.h(o)

    def comp():
        for j, cl in enumerate(problem.clauses):
            neg = [x[l - 1] for l in cl if l > 0]
            if neg: qc.x(neg)
            qc.mcx([x[abs(l) - 1] for l in cl], u[j])
            if neg: qc.x(neg)
        off = len(problem.clauses)
        for j, (vs, rhs) in enumerate(problem.xors):
            for v in vs: qc.cx(x[v - 1], u[off + j])
            if rhs == 1: qc.x(u[off + j])
    for _ in range(iterations):
        comp(); qc.x(u); qc.mcx(list(u), o[0]); qc.x(u); comp()
        qc.h(x); qc.x(x); qc.h(x[-1]); qc.mcx(list(x[:-1]), x[-1]); qc.h(x[-1]); qc.x(x); qc.h(x)
    qc.measure(x, out)
    return qc


# ----------------------------------------------------------------------------------------
# Analysis helpers
# ----------------------------------------------------------------------------------------
def optimal_iterations(N, M):
    return max(0, int(math.floor(math.pi / (4 * math.asin(math.sqrt(M / N))))))


def herald_model(N, M, T, c=5.0):
    """Exact 2-D model of the heralded search. Returns per-step click probabilities and
    the probability that the final (unheralded) measurement is a solution."""
    th = math.asin(math.sqrt(M / N))
    u = np.array([math.sin(th), math.cos(th)])
    v = u.copy()
    clicks = []
    for t in range(T):
        phi = math.asin(math.sqrt(min(0.5, c / (t + 1))))
        v = np.array([-v[0], v[1]])
        clicks.append((v[0] * math.sin(phi)) ** 2)
        v = np.array([v[0] * math.cos(phi), v[1]])
        v = 2 * (u @ v) * u - v
    return np.array(clicks), v[0] ** 2


def expected_queries_herald(N, M, c=5.0, T=None):
    """Repeat-until-success expected oracle calls (unheralded rounds end with one verification query)."""
    T = T or int(40 * math.sqrt(N / M)) + 80
    clicks, pg = herald_model(N, M, T, c)
    pend = 1 - clicks.sum()
    cost = (clicks * np.arange(1, T + 1)).sum() + pend * (T + 1)
    return cost / (clicks.sum() + pg)


def expected_queries_grover_known_M(N, M):
    k = optimal_iterations(N, M)
    th = math.asin(math.sqrt(M / N))
    return (k + 1) / math.sin((2 * k + 1) * th) ** 2


def expected_queries_bbht(N, M, trials=20000, lam=6 / 5, seed=1):
    rng = np.random.default_rng(seed)
    th = math.asin(math.sqrt(M / N))
    tot = 0
    for _ in range(trials):
        mm, cnt = 1.0, 0
        while True:
            j = int(rng.integers(0, int(math.ceil(mm))))
            cnt += j + 1
            if rng.random() < math.sin((2 * j + 1) * th) ** 2:
                break
            mm = min(lam * mm, math.sqrt(N))
        tot += cnt
    return tot / trials


def expected_cz(qc, branch_prob=0.5):
    """Count CZ gates; gates inside if_else bodies are weighted by branch_prob (MBU branches)."""
    total, worst = 0.0, 0
    for inst in qc.data:
        op = inst.operation
        if op.name == "if_else":
            e, w = expected_cz(op.params[0], branch_prob)
            total += branch_prob * e
            worst += w
        elif op.name in ("cz", "cx", "ecr"):
            total += 1; worst += 1
    return total, worst


def split_counts(counts, qc):
    """Turn Aer counts into a list of ({regname: bits}, count)."""
    names = [c.name for c in qc.cregs][::-1]
    out = []
    for b, c in counts.items():
        parts = b.split()
        out.append((dict(zip(names, parts)), c))
    return out


def click_iteration(hbits: str):
    """Index (0-based) of the iteration whose herald clicked, or None. hbits is Qiskit order."""
    r = hbits[::-1]
    return r.index("1") if "1" in r else None
