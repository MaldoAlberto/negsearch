r"""Clifford+T-level circuits (ccx, ch, cx, x, z) for negation-forced amplitude amplification.

A      : in-place preparation; each variable is its own coin (H only if not forced)
O      : phase oracle  (-1)^{F(x)}
S0     : reflection about |0...0> on the variable register
Q      : A S0 A^dagger O      (amplitude amplification about A|0>)
Baseline Grover = same with A = H^{\otimes n}.
All work ancillas are returned to |0> and reused from a pool.
"""
from __future__ import annotations

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from .core import CNF, forcing_plan


class _Pool:
    def __init__(self, qubits):
        self.free = list(qubits)
        self.min_free = len(self.free)

    def get(self):
        q = self.free.pop()
        self.min_free = min(self.min_free, len(self.free))
        return q

    def put(self, q):
        self.free.append(q)


def and_into(qc, pool, ctrls, on_zero, tgt):
    """tgt ^= AND_i [ctrl_i == (0 if on_zero_i else 1)]  using only ccx and a chain of temps."""
    flips = [q for q, z in zip(ctrls, on_zero) if z]
    if flips:
        qc.x(flips)
    if len(ctrls) == 1:
        qc.cx(ctrls[0], tgt)
    else:
        temps, prev = [], ctrls[0]
        for c in ctrls[1:-1]:
            t = pool.get()
            qc.ccx(prev, c, t)
            temps.append((prev, c, t))
            prev = t
        qc.ccx(prev, ctrls[-1], tgt)
        for p, c, t in reversed(temps):
            qc.ccx(p, c, t)
            pool.put(t)
    if flips:
        qc.x(flips)


class NegForcedSearch:
    def __init__(self, cnf: CNF, order=None, K=None, baseline=False):
        self.cnf, self.K, self.baseline = cnf, K, baseline
        self.order = list(order) if order is not None else list(range(1, cnf.n + 1))
        self.plan = forcing_plan(cnf, self.order, K)
        n, m = cnf.n, cnf.m
        kmax = max(len(c) for c in cnf.clauses)
        fmax = max(len(a) + len(b) for _, a, b in self.plan) if self.plan else 0
        need = max(2 * fmax + 4, kmax, m, n + 1) + 2
        self.x = QuantumRegister(n, "x")
        self.u = QuantumRegister(m, "u")
        self.w = QuantumRegister(need, "w")
        self.f = QuantumRegister(1, "f")
        self.c = ClassicalRegister(n, "out")
        self.qc = QuantumCircuit(self.x, self.u, self.w, self.f, self.c)
        self.pool = _Pool(self.w)

    # -------------------------------------------------------------------------------
    def _prep_body(self):
        qc = QuantumCircuit(*self.qc.qregs)
        x, pool = self.x, self.pool
        if self.baseline:
            qc.h(x)
            return qc
        for v, C1, C0 in self.plan:
            anc = []
            for cl in C1 + C0:  # a_c = all other literals of clause cl are FALSE
                t = pool.get()
                others = [l for l in cl if abs(l) != v]
                and_into(qc, pool, [x[abs(l) - 1] for l in others], [l > 0 for l in others], t)
                anc.append(t)
            f1, f0, g = pool.get(), pool.get(), pool.get()

            def orr(lst, t):
                if lst:
                    and_into(qc, pool, lst, [True] * len(lst), t)
                    qc.x(t)

            a1, a0 = anc[: len(C1)], anc[len(C1):]
            orr(a1, f1)
            orr(a0, f0)
            qc.cx(f1, x[v - 1])                                   # forced to 1
            and_into(qc, pool, [f1, f0], [True, True], g)        # g = not forced
            qc.ch(g, x[v - 1])                                    # decision (coin = variable)
            and_into(qc, pool, [f1, f0], [True, True], g)
            orr(a0, f0)
            orr(a1, f1)
            for q in (g, f0, f1):
                pool.put(q)
            for cl, t in reversed(list(zip(C1 + C0, anc))):
                others = [l for l in cl if abs(l) != v]
                and_into(qc, pool, [x[abs(l) - 1] for l in others], [l > 0 for l in others], t)
                pool.put(t)
        return qc

    def prep(self, inverse=False):
        body = self._prep_body()
        self.qc.compose(body.inverse() if inverse else body, inplace=True)

    def oracle(self):
        qc, x, u, pool = self.qc, self.x, self.u, self.pool
        def compute():
            for j, cl in enumerate(self.cnf.clauses):   # u_j = clause j violated
                and_into(qc, pool, [x[abs(l) - 1] for l in cl], [l > 0 for l in cl], u[j])
        compute()
        and_into(qc, pool, list(u), [True] * len(u), self.f[0])
        qc.z(self.f[0])
        and_into(qc, pool, list(u), [True] * len(u), self.f[0])
        compute()

    def reflect_zero(self):
        qc, xs, pool = self.qc, list(self.x), self.pool
        if len(xs) == 1:
            qc.x(xs[0]); qc.z(xs[0]); qc.x(xs[0])
            return
        t = pool.get()
        and_into(qc, pool, xs[:-1], [True] * (len(xs) - 1), t)
        qc.x(xs[-1]); qc.cz(t, xs[-1]); qc.x(xs[-1])
        and_into(qc, pool, xs[:-1], [True] * (len(xs) - 1), t)
        pool.put(t)

    def iterate(self):
        self.oracle()
        self.prep(inverse=True)
        self.reflect_zero()
        self.prep()

    def amplified(self, k, measure=True):
        self.prep()
        for _ in range(k):
            self.iterate()
        if measure:
            self.qc.measure(self.x, self.c)
        return self.qc


def component_circuits(cnf, order, K=None):
    """Separate circuits for resource accounting: A, O, S0 (each without measurements)."""
    out = {}
    for name in ("A", "O", "S0"):
        b = NegForcedSearch(cnf, order, K)
        if name == "A":
            b.prep()
        elif name == "O":
            b.oracle()
        else:
            b.reflect_zero()
        qc = b.qc.copy()
        qc.remove_final_measurements()
        out[name] = qc
    return out


SIM_BASIS = ["ccx", "cx", "cz", "h", "x", "z", "s", "sdg", "t", "tdg", "u", "measure", "reset"]


def for_simulation(qc):
    """Decompose controlled-H etc. for Aer (no coupling map, any width)."""
    from qiskit import transpile
    return transpile(qc, basis_gates=SIM_BASIS, optimization_level=0)
