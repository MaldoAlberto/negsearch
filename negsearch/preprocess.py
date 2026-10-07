"""Classical preprocessing of a constraint set BEFORE the circuit is generated.  All passes are exact (the feasible set is unchanged
up to the removed forced variables), so feasibility by construction is preserved.
  clean_clauses       dedupe, tautologies, subsumption, unit propagation (forced variables stop being qubits)
  automaton_from_clauses(order)   reduced automaton of a clause set under a variable order (state = frontier assignment)
  search_order        hill climbing on adjacent swaps (+ degree / degeneracy / colouring starts) to minimise the automaton width
  detect_unit_symmetry  groups of units (tuples of variables) that can be exchanged without leaving F  -> counting structure
Clauses are tuples of non-zero ints: +v+1 = x_v, -(v+1) = NOT x_v (a clause is violated iff every literal is false)."""
from __future__ import annotations
import itertools
from .structural import build_structural
from .depth_tools import coloring_layer_order, degeneracy_order


def clean_clauses(clauses, n):
    cl = {frozenset(c) for c in clauses}
    cl = {c for c in cl if not any(-l in c for l in c)}
    fixed = {}
    changed = True
    while changed:
        changed = False
        for c in list(cl):
            if len(c) == 0:
                raise ValueError('infeasible')
            if len(c) == 1:
                (l,) = tuple(c); v = abs(l) - 1; val = 1 if l > 0 else 0
                if v in fixed and fixed[v] != val: raise ValueError('infeasible')
                fixed[v] = val; changed = True
        if changed:
            new = set()
            for c in cl:
                if any((abs(l) - 1) in fixed and (fixed[abs(l) - 1] == (1 if l > 0 else 0)) for l in c): continue     # satisfied
                new.add(frozenset(l for l in c if (abs(l) - 1) not in fixed))
            cl = new
    keep = []
    for c in sorted(cl, key=len):
        if not any(k <= c for k in keep): keep.append(c)          # subsumption
    live = sorted({abs(l) - 1 for c in keep for l in c})
    return [tuple(sorted(c)) for c in keep], fixed, live


def automaton_from_clauses(n, clauses, order):
    pos = {v: k for k, v in enumerate(order)}
    last = {i: max(pos[abs(l) - 1] for l in c) for i, c in enumerate(clauses)}          # level at which clause becomes fully assigned
    touch = {v: [i for i, c in enumerate(clauses) if any(abs(l) - 1 == v for l in c)] for v in range(n)}

    def step(k, st, b):
        v = order[k]; a = dict(st); a[v] = b
        for i in touch[v]:
            if last[i] == k and all(a[abs(l) - 1] == (0 if l > 0 else 1) for l in clauses[i]): return None          # violated
        keep = {u: val for u, val in a.items() if any(last[i] > k for i in touch[u])}                                # frontier
        return tuple(sorted(keep.items()))
    return build_structural(n, (), step, lambda st: True)


def _adj(n, clauses):
    N = [set() for _ in range(n)]
    for c in clauses:
        vs = [abs(l) - 1 for l in c]
        for a, b in itertools.combinations(vs, 2): N[a].add(b); N[b].add(a)
    return N


def search_order(n, clauses, iters=300, seed=0):
    import random
    rnd = random.Random(seed); N = _adj(n, clauses)
    starts = {'index': list(range(n)), 'degree': sorted(range(n), key=lambda v: (len(N[v]), v)),
              'degeneracy': degeneracy_order(N, n), 'colouring': coloring_layer_order(N, n)}
    w = {k: automaton_from_clauses(n, clauses, o).w for k, o in starts.items()}
    bk_ = min(w, key=w.get); best = list(starts[bk_]); bw = w[bk_]
    for _ in range(iters):
        i = rnd.randrange(n - 1); o = best[:]; o[i], o[i + 1] = o[i + 1], o[i]
        ww = automaton_from_clauses(n, clauses, o).w
        if ww <= bw: best, bw = o, ww
    return best, bw, w


def detect_unit_symmetry(F, units):
    """F: set of tuples (feasible strings); units: list of tuples of variable indices of equal length.  Returns classes of units that are
    pairwise interchangeable (swapping their values maps F onto F)."""
    F = set(F); m = len(units); parent = list(range(m))
    def find(a):
        while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for a, b in itertools.combinations(range(m), 2):
        ua, ub = units[a], units[b]
        ok = True
        for x in F:
            y = list(x)
            for i, j in zip(ua, ub): y[i], y[j] = x[j], x[i]
            if tuple(y) not in F: ok = False; break
        if ok: parent[find(a)] = find(b)
    classes = {}
    for a in range(m): classes.setdefault(find(a), []).append(a)
    return sorted(classes.values())
