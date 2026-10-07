"""Exact classical preprocessing before circuit generation: clause cleaning, width-minimising order, unit-symmetry detection and lowering.
Writes results/preprocess_study.json."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)
import json, random, warnings; warnings.filterwarnings('ignore')
import numpy as np
from qiskit import transpile
from negsearch.preprocess import clean_clauses, automaton_from_clauses, search_order, detect_unit_symmetry
from negsearch.onehot_rc import compile_Aq_onehot_rc
from negsearch.symlower import symmetric_block_prep
from negsearch.hw_large import block_strings
from negsearch.large_case import sector_block_automaton
from negsearch.fez_check import bk


def cz(qc, seeds=3):
    return min(transpile(qc, bk, optimization_level=3, seed_transpiler=s + 1).count_ops().get('cz', 0) for s in range(seeds))


out = {}
# ---- 1. clause cleaning + order search on conflict-type constraints (independent set / set packing with redundant clauses and forced vars)
rows = []
for n, p, seed in [(14, 0.25, 1), (18, 0.2, 2), (22, 0.15, 3), (26, 0.12, 4)]:
    rnd = random.Random(seed)
    cl = [(-(a + 1), -(b + 1)) for a in range(n) for b in range(a + 1, n) if rnd.random() < p]
    raw = list(cl)
    raw += [c for c in cl[: len(cl) // 3]]                                                     # repeated clauses
    raw += [(-(a + 1), -(b + 1), -(c + 1)) for (a, b, c) in [(0, 1, 2), (3, 4, 5)] if False]
    raw += [(-(rnd.randrange(n) + 1),)] * 2                                                   # two forced-to-0 variables (unit clauses)
    raw += [tuple(sorted(set(c + (-(rnd.randrange(n) + 1),)))) for c in cl[:6]]               # subsumed clauses (supersets)
    clean, fixed, live = clean_clauses(raw, n)
    order0 = list(range(n)); w_raw = automaton_from_clauses(n, [tuple(c) for c in raw if len(c) > 1], order0).w
    # work on the live variables only
    idx = {v: k for k, v in enumerate(live)}
    cl2 = [tuple((1 if l > 0 else -1) * (idx[abs(l) - 1] + 1) for l in c) for c in clean]
    n2 = len(live)
    w_clean = automaton_from_clauses(n2, cl2, list(range(n2))).w
    best, w_best, ws = search_order(n2, cl2)
    rows.append(dict(n=n, clauses_raw=len(raw), clauses_clean=len(clean), forced=len(fixed), n_live=n2, width_raw_index=w_raw, width_clean_index=w_clean,
                     width_orders=ws, width_best=w_best))
    print(f"MIS-like n={n}: clauses {len(raw)}->{len(clean)}, forced vars {len(fixed)}, qubits {n}->{n2}; width {w_raw} -> {w_clean} (index order) -> {w_best} (searched); orders {ws}", flush=True)
out['clauses'] = rows

# ---- 2. unit symmetry in the portfolio blocks and its lowering
rows = []
for m, (l, s) in [(3, (1, 1)), (4, (1, 1)), (4, (1, 2)), (5, (0, 3)), (5, (1, 1)), (5, (1, 2))]:
    B = sector_block_automaton(m, l, s)
    F = [x for x, _ in block_strings(B)]
    units = [(2 * j, 2 * j + 1) for j in range(m)]
    cls = detect_unit_symmetry(F, units)
    g = compile_Aq_onehot_rc(B, uncompute=False)[0]; gu = compile_Aq_onehot_rc(B, uncompute=True)[0]
    sym = symmetric_block_prep(m, l, s)
    r = dict(m=m, l=l, s=s, F=len(F), classes=[len(c) for c in cls], generic_qubits=g.num_qubits, generic_cz=cz(g, 2), generic_cz_uncomputed=cz(gu, 2), sym_qubits=sym.num_qubits, sym_cz=cz(sym))
    rows.append(r)
    print(f"block m={m} (l,s)=({l},{s}): |F|={len(F)}, interchangeable-unit classes {r['classes']}; generic A_q {r['generic_cz']} CZ ({r['generic_cz_uncomputed']} uncomputed, {r['generic_qubits']} q) -> symmetric lowering {r['sym_cz']} CZ ({r['sym_qubits']} q)", flush=True)
out['blocks'] = rows
json.dump(out, open('results/preprocess_study.json', 'w'), indent=1)
