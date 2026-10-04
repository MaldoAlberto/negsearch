"""Problem families, negation-forced decision planner and exact success probabilities.

A CNF clause (l1 v l2 v ... v lk) is the negation of a forbidden pattern.  Reading it as a
*negative condition*: if every literal except l_v is already false, then l_v is forced.
A variable that is forced is set deterministically; otherwise it is a decision (a Hadamard).
"""
from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field

from pysat.solvers import Minisat22


# ------------------------------------------------------------------------------------------
# Problems
# ------------------------------------------------------------------------------------------
@dataclass
class CNF:
    n: int
    clauses: list
    family: str = "cnf"
    meta: dict = field(default_factory=dict)

    @property
    def m(self):
        return len(self.clauses)

    def satisfied(self, a):  # a: dict var->0/1 or list indexed var-1
        get = (lambda v: a[v]) if isinstance(a, dict) else (lambda v: a[v - 1])
        return all(any((get(abs(l)) == 1) == (l > 0) for l in c) for c in self.clauses)

    def check_bitstring(self, s):  # qiskit order, rightmost = variable 1
        bits = [int(ch) for ch in s[::-1]]
        return self.satisfied(bits)


def count_models(cnf: CNF, cap=200_000):
    return len(enumerate_models(cnf, cap))


def random_ksat(n, k, alpha, rng):
    m = int(round(alpha * n))
    cls = [tuple(v * (1 if rng.random() < 0.5 else -1) for v in rng.sample(range(1, n + 1), k))
           for _ in range(m)]
    return CNF(n, cls, f"{k}-SAT", dict(alpha=alpha))


def one_in_three_sat(n, alpha, rng):
    """Positive 1-in-3 SAT (exact-cover-like): each triple has exactly one true variable."""
    m = int(round(alpha * n))
    cls = []
    for _ in range(m):
        a, b, c = rng.sample(range(1, n + 1), 3)
        cls += [(a, b, c), (-a, -b), (-a, -c), (-b, -c)]
    return CNF(n, cls, "1-in-3-SAT", dict(alpha=alpha, triples=m))


def graph_3col(nv, avg_deg, rng):
    """3-colouring of an Erdos-Renyi graph, 2 bits per vertex (pattern 11 forbidden)."""
    var = lambda v, b: 2 * v + b + 1
    edges = set()
    target = min(int(round(avg_deg * nv / 2)), nv * (nv - 1) // 2 - 1)
    while len(edges) < target:
        u, w = rng.sample(range(nv), 2)
        edges.add((min(u, w), max(u, w)))
    cls = [(-var(v, 0), -var(v, 1)) for v in range(nv)]
    for u, w in sorted(edges):
        for col in range(3):
            b0, b1 = col & 1, (col >> 1) & 1
            lit = lambda x, b, val: -var(x, b) if val else var(x, b)
            cls.append((lit(u, 0, b0), lit(u, 1, b1), lit(w, 0, b0), lit(w, 1, b1)))
    return CNF(2 * nv, cls, "3-COL", dict(vertices=nv, avg_deg=avg_deg, edges=len(edges)))


FAMILIES = {
    "3-SAT": dict(make=lambda n, rng: random_ksat(n, 3, 4.2, rng)),
    "4-SAT": dict(make=lambda n, rng: random_ksat(n, 4, 9.6, rng)),
    "1-in-3-SAT": dict(make=lambda n, rng: one_in_three_sat(n, 0.6, rng)),
    "3-COL": dict(make=lambda n, rng: graph_3col(n // 2, 3.0 if n // 2 < 10 else 4.2, rng)),
}


def sample_satisfiable(family, n, rng, cap=200_000, max_tries=400):
    for _ in range(max_tries):
        f = FAMILIES[family]["make"](n, rng)
        M = count_models(f, cap)
        if 0 < M < cap:
            f.meta["M"] = M
            return f
    raise RuntimeError(f"no satisfiable {family} instance with n={n}")


# ------------------------------------------------------------------------------------------
# Negation-forced decision planner
# ------------------------------------------------------------------------------------------
def forcing_plan(cnf: CNF, order, K=None):
    """For each variable v (in order): clauses that can force v=1 (C1) or v=0 (C0) because all
    their other variables precede v.  K caps the number of conditions checked per variable."""
    pos = {v: i for i, v in enumerate(order)}
    plan = []
    for v in order:
        C1, C0 = [], []
        for c in cnf.clauses:
            if any(abs(l) == v for l in c) and all(pos[abs(l)] < pos[v] for l in c if abs(l) != v):
                (C1 if v in c else C0).append(c)
        if K is not None:
            C1, C0 = C1[:K], C0[:K]
        plan.append((v, C1, C0))
    return plan


def _forced(plan_entry, a):
    v, C1, C0 = plan_entry
    f = lambda c: all((a[abs(l)] == 1) != (l > 0) for l in c if abs(l) != v)
    return any(f(c) for c in C1), any(f(c) for c in C0)


def exact_success(cnf: CNF, order, K=None, max_leaves=1 << 26):
    """Exact probability that the in-place forced preparation A|0> yields a solution.
    Returns (p, expected_decisions_on_successful_paths_weighted)."""
    plan = forcing_plan(cnf, order, K)
    n = cnf.n
    total = 0.0
    leaves = 0

    def rec(i, a, w):
        nonlocal total, leaves
        if i == n:
            leaves += 1
            if leaves > max_leaves:
                raise RuntimeError("too many leaves")
            if cnf.satisfied(a):
                total += w
            return
        v = plan[i][0]
        f1, f0 = _forced(plan[i], a)
        if f1 or f0:
            a[v] = 1 if f1 else 0
            rec(i + 1, a, w)
        else:
            for b in (0, 1):
                a[v] = b
                rec(i + 1, a, w / 2)
        del a[v]

    rec(0, {}, 1.0)
    return total


def decisions_along(cnf: CNF, order, solution, K=None):
    """D_pi(a*): number of non-forced variables when following solution a* (Lemma 1)."""
    plan = forcing_plan(cnf, order, K)
    a, D = {}, 0
    for entry in plan:
        f1, f0 = _forced(entry, a)
        v = entry[0]
        if not (f1 or f0):
            D += 1
        a[v] = solution[v - 1]
    return D


def amplification_iterations(p):
    th = math.asin(math.sqrt(p))
    return max(0, int(math.floor(math.pi / (4 * th))))


def amplified_success(p, k):
    return math.sin((2 * k + 1) * math.asin(math.sqrt(p))) ** 2


def enumerate_models(cnf: CNF, cap=200_000):
    s = Minisat22(bootstrap_with=[list(c) for c in cnf.clauses])
    sols = []
    while len(sols) < cap and s.solve():
        mod = s.get_model()[: cnf.n]
        base = [1 if l > 0 else 0 for l in mod]
        tail = cnf.n - len(base)          # variables above the solver's max var are free
        for bits in itertools.product((0, 1), repeat=tail):
            sols.append(base + list(bits))
        s.add_clause([-l for l in mod])
    s.delete()
    return sols[:cap] if len(sols) <= cap else sols


def success_by_solutions(cnf: CNF, order, sols, K=None):
    """Lemma 3 of the paper (exact): a forced value never contradicts a satisfying assignment, hence
    p_pi = sum_{a in Sol} 2^{-D_pi(a)}.  Polynomial per solution."""
    plan = forcing_plan(cnf, order, K)
    tot, Ds = 0.0, []
    for a in sols:
        D = decisions_along_plan(plan, a)
        Ds.append(D)
        tot += 2.0 ** (-D)
    return tot, Ds


def decisions_along_plan(plan, solution):
    a, D = {}, 0
    for entry in plan:
        f1, f0 = _forced(entry, a)
        if not (f1 or f0):
            D += 1
        a[entry[0]] = solution[entry[0] - 1]
    return D
