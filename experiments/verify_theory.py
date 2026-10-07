"""Do the theoretical guarantees survive the optimisations of the large case?  Three checks.
 (1) circuit level: symmetric preparation + objective phase (full or pruned) + pair-exchange mixers, simulated with Qiskit's statevector at random
     angles and p=1..3, must put probability 1 on the feasible set of the interface tuple and agree with an independent dense numerical simulation;
 (2) amplification (Theorem 2 / Remark 1): iterations over the standard Grover state, the coherent A_q (coin walk) and the uniform state on F; and what
     conditioning on a classically drawn interface costs (the coherence it gives up);
 (3) preprocessing (cleaning, unit propagation, variable order) leaves the feasible set unchanged.
Writes results/verify_theory.json."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)
import json, math, itertools, random, warnings; warnings.filterwarnings('ignore')
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from negsearch.hw_large import interface_tuples, qubo_cost, ising_from_qubo, prune, x_index, swap_mixer_layer, swap_pairs, block_strings
from negsearch.depth_study import phase_poly
from negsearch.symlower import symmetric_block_prep
from negsearch.large_case import sector_block_automaton

OUT = {}


# ------------------------------------------------------------------ (1)
def build(A, T, S, iface, co, gam, bet):
    nx = 2 * A * T; qc = QuantumCircuit(nx); off = np.cumsum([0] + list(S)); blocks = []
    for t in range(T):
        for si, size in enumerate(S):
            l, s = iface[t][si]; xq = [x_index(A, t, off[si] + j, d) for j in range(size) for d in (0, 1)]
            qc.compose(symmetric_block_prep(size, l, s), qubits=xq, inplace=True); blocks.append(xq)
    for g, b in zip(gam, bet):
        phase_poly(qc, co, g)
        for xq in blocks: swap_mixer_layer(qc, xq, b, pairs='path')
    return qc, blocks


def dense_reference(A, T, S, iface, co, gam, bet):
    """independent dense simulation: uniform state on F(iface), phase from the Ising polynomial, pair-exchange rotations cos b I - i sin b P"""
    nx = 2 * A * T; off = np.cumsum([0] + list(S)); N = 2 ** nx
    idx = np.arange(N); bits = ((idx[:, None] >> np.arange(nx)) & 1)
    feas = np.ones(N, bool); blocks = []
    for t in range(T):
        for si, size in enumerate(S):
            l, s = iface[t][si]; xq = [x_index(A, t, off[si] + j, d) for j in range(size) for d in (0, 1)]; blocks.append((xq, size))
            L = bits[:, xq[0::2]].sum(1); Sh = bits[:, xq[1::2]].sum(1); ex = (bits[:, xq[0::2]] & bits[:, xq[1::2]]).sum(1)
            feas &= (L == l) & (Sh == s) & (ex == 0)
    psi = feas / math.sqrt(feas.sum()); psi = psi.astype(complex)
    z = 1 - 2 * bits; ph = np.zeros(N)
    for k, v in co.items(): ph += v * np.prod(z[:, list(k)], axis=1)
    for g, b in zip(gam, bet):
        psi = psi * np.exp(-1j * g * ph)
        for xq, size in blocks:
            for j, k in swap_pairs(size, 'path'):
                perm = idx.copy()
                for a_, b_ in ((xq[2 * j], xq[2 * k]), (xq[2 * j + 1], xq[2 * k + 1])):
                    ba = (perm >> a_) & 1; bb = (perm >> b_) & 1; same = ba == bb
                    perm = np.where(same, perm, perm ^ ((1 << a_) | (1 << b_)))
                psi = math.cos(b) * psi - 1j * math.sin(b) * psi[perm]
    return psi, feas


rng = np.random.default_rng(7); rows = []
for A, T, S in [(4, 1, [2, 2]), (6, 1, [3, 3]), (4, 2, [2, 2])]:
    n = 2 * A * T; h, J = qubo_cost(A, T, 5); co_full = ising_from_qubo(h, J)
    tup, _ = interface_tuples(A, T, S); iface = max(tup, key=lambda x: x[1])[0]
    for name, co in (('full', co_full), ('pruned 25%', prune(co_full, 0.25))):
        worst = 0.0; maxdiff = 0.0
        for p in (1, 2, 3):
            for rep in range(4):
                gam = rng.uniform(-2, 2, p); bet = rng.uniform(0, 3, p)
                qc, _ = build(A, T, S, iface, co, gam, bet)
                psi = Statevector(qc).data
                ref, feas = dense_reference(A, T, S, iface, co, gam, bet)
                worst = max(worst, 1 - float(np.abs(psi[feas]).__pow__(2).sum()))
                ov = abs(np.vdot(ref, psi)); maxdiff = max(maxdiff, 1 - ov)
        rows.append(dict(n=n, objective=name, interface=str(iface), max_infeasible_mass=worst, max_one_minus_overlap=maxdiff))
        print(f"(1) n={n:2d} objective {name:10s}: largest infeasible probability over 12 random angle sets (p=1..3) {worst:.1e}; 1-|<reference|circuit>| = {maxdiff:.1e}", flush=True)
OUT['circuit'] = rows

# ------------------------------------------------------------------ (2)
import experiments.large_case_quality as LQ
from negsearch.large_case import case_automaton
Aut, var = case_automaton(4, 2, [2, 2], 1, 3, 3, int(round(0.6 * 2 * 3)), 'period')
def enum(Aut):
    out = []
    def rec(k, s, x):
        if k == Aut.n: out.append(tuple(x)); return
        t0, t1 = Aut.trans[k][s]
        if t0 >= 0: rec(k + 1, t0, x + [0])
        if t1 >= 0: rec(k + 1, t1, x + [1])
    rec(0, 0, []); return out
strs = enum(Aut)
n = Aut.n; Fset = set(strs)
LQset = {tuple(int(v) for v in row) for row in LQ.X[LQ.FEAS]}
print('(2) consistency of the automaton with the exact feasible set used for the numerics:', Fset == LQset, len(Fset), len(LQset))
OUT['automaton_matches_F'] = bool(Fset == LQset)
# coin-walk probabilities mu(x) = 2^-D(x)
def mu_of(Aut, x):
    s = 0; D = 0
    for k, b in enumerate(x):
        t0, t1 = Aut.trans[k][s]
        if t0 >= 0 and t1 >= 0: D += 1
        s = t1 if b else t0
    return 2.0 ** (-D)
mu = {x: mu_of(Aut, x) for x in Fset}
print('    sum of mu over F (must be 1):', round(sum(mu.values()), 12))
def iters(a):
    return math.floor(math.pi / (4 * math.asin(math.sqrt(min(a, 1.0)))))
interface = {x: tuple(sum(x[LQ.idx(t, i, d)] for i in range(LQ.A) for d in (0, 1)) for t in range(LQ.T)) for x in Fset}
groups = {}
for x in Fset: groups.setdefault(interface[x], []).append(x)
F_size = len(Fset); rows = []
for seed in (1, 2):
    c = LQ.make_cost(seed); cd = {tuple(int(v) for v in row): c[i] for i, row in enumerate(LQ.X) if tuple(int(v) for v in row) in Fset}
    order = sorted(Fset, key=lambda x: cd[x])
    for gsize in (1, 5, 26):                      # the best 1 string, best 5, best 5% of F
        G = order[:gsize]; a0 = gsize / 2 ** n; aF = gsize / F_size; aq = sum(mu[x] for x in G)
        k0, kF, kq = iters(a0), iters(aF), iters(aq)
        # conditioning: draw the interface with weight |group|/|F|, amplify inside the group (the cost of a full run until success)
        cost = 0.0
        gs = {}
        for x in G: gs.setdefault(interface[x], []).append(x)
        tot_w = sum(len(groups[g]) / F_size for g in gs)
        # best case: the group holding the largest part of G is known to be the one to amplify; expected iterations = k_g / P(draw that group)
        g_star = max(gs, key=lambda g: len(gs[g])); w = len(groups[g_star]) / F_size; ag = len(gs[g_star]) / len(groups[g_star])
        kc = iters(ag) / w
        rows.append(dict(seed=seed, G=gsize, a_standard=a0, a_uniformF=aF, a_coin=aq, k_standard=k0, k_uniformF=kF, k_coin=kq, k_conditioned=kc, w_group=w, samples_classical=1 / aF))
        print(f"(2) seed {seed}, |G|={gsize:2d}: iterations standard Grover {k0}, A_q coin walk {kq}, uniform on F {kF}; conditioned on the interface {kc:.0f} (group weight {w:.3f}); classical sampling of F {1/aF:.0f}", flush=True)
OUT['amplification'] = rows

# ------------------------------------------------------------------ (3)
from negsearch.preprocess import clean_clauses, automaton_from_clauses, search_order
rows = []; ok = True
for seed in range(8):
    rnd = random.Random(seed); n = rnd.choice([9, 10, 11, 12]); p = rnd.choice([0.2, 0.3])
    cl = [(-(a + 1), -(b + 1)) for a in range(n) for b in range(a + 1, n) if rnd.random() < p]
    cl += cl[: len(cl) // 3] + [(-(rnd.randrange(n) + 1),)] + [tuple(sorted(set(c + (-(rnd.randrange(n) + 1),)))) for c in cl[:4]] + [((rnd.randrange(n) + 1), -(rnd.randrange(n) + 1))]
    cl = [c for c in cl if c]
    def sat(x, clauses): return all(any((x[abs(l) - 1] == 1) == (l > 0) for l in c) for c in clauses)
    F = {x for x in itertools.product((0, 1), repeat=n) if sat(x, cl)}
    clean, fixed, live = clean_clauses(cl, n)
    # F recovered from the cleaned instance: forced variables fixed, free variables not in any clause unconstrained
    Fc = set()
    for xl in itertools.product((0, 1), repeat=len(live)):
        x = [None] * n
        for v, val in fixed.items(): x[v] = val
        for v, val in zip(live, xl): x[v] = val
        free = [i for i in range(n) if x[i] is None]
        for fv in itertools.product((0, 1), repeat=len(free)):
            y = list(x)
            for i, val in zip(free, fv): y[i] = val
            if sat(y, clean) and all(y[v] == val for v, val in fixed.items()): Fc.add(tuple(y))
    same_clean = (F == Fc)
    # automaton under several orders enumerates the same set
    cl2 = [tuple(c) for c in cl if len(c) > 1]
    orders = {'index': list(range(n)), 'reverse': list(range(n))[::-1], 'random': random.Random(seed).sample(range(n), n)}
    cl_nz = [c for c in cl]
    bo, bw, _ = search_order(n, [c for c in cl if len(c) > 1], iters=40, seed=seed); orders['searched'] = bo
    same_aut = True
    for nm, o in orders.items():
        Au = automaton_from_clauses(n, [c for c in cl if len(c) > 1], o)
        E = set()
        def rec(k, s, x):
            if k == Au.n:
                y = [0] * n
                for kk, v in enumerate(o): y[v] = x[kk]
                E.add(tuple(y)); return
            t0, t1 = Au.trans[k][s]
            if t0 >= 0: rec(k + 1, t0, x + [0])
            if t1 >= 0: rec(k + 1, t1, x + [1])
        rec(0, 0, [])
        Fsub = {x for x in itertools.product((0, 1), repeat=n) if sat(x, [c for c in cl if len(c) > 1])}
        same_aut &= (E == Fsub)
    ok &= same_clean and same_aut
    rows.append(dict(seed=seed, n=n, F=len(F), cleaning_preserves_F=same_clean, automaton_same_F_all_orders=same_aut))
    print(f"(3) instance {seed}: n={n}, |F|={len(F)}, cleaning preserves F: {same_clean}; automaton enumerates the same set under 4 orders: {same_aut}", flush=True)
OUT['preprocessing'] = rows; OUT['preprocessing_all_ok'] = bool(ok)
json.dump(OUT, open('results/verify_theory.json', 'w'), indent=1, default=float)
