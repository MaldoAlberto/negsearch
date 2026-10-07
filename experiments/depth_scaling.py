"""Circuit depth / two-qubit gate count vs number of variables for each method.
MIS: BFS-ball induced subgraphs of the QOBLIB 'karate' (34) and 'chesapeake' (39) graphs.
3-SAT: negation-forced preparation and Grover iteration from the paper's construction."""
import os as _os, sys as _sys
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT); _sys.path.insert(0, ROOT)
import json, random, time
import numpy as np
from negsearch.qaoa_mis import read_gph
from negsearch.qaoa_circuits import prep_circuit, cg_qaoa, penalty_qaoa, hadfield_qaoa, cg_qaoa_anc, prep_circuit_anc
from negsearch.core import sample_satisfiable
from negsearch.circuits import NegForcedSearch
from qiskit import transpile, QuantumCircuit
from qiskit_ibm_runtime.fake_provider import FakeFez
fake = FakeFez(); LOG = ['cx', 'rz', 'sx', 'x']
def counts(qc, routed):
    t = transpile(qc, fake, optimization_level=1, seed_transpiler=1) if routed else transpile(qc, basis_gates=LOG, optimization_level=1)
    two = sum(v for k, v in t.count_ops().items() if k in ('cx', 'cz', 'ecr'))
    return dict(depth=t.depth(), twoq=two)
def degeneracy_order(N, n):
    rem = set(range(n)); seq = []
    while rem:
        v = min(rem, key=lambda u: (len(N[u] & rem), u)); seq.append(v); rem.remove(v)
    return seq[::-1]
out = dict(mis=[], sat=[])
Q = _os.environ.get('QOBLIB_MIS', 'qoblib/07-independentset/') + 'instances/'
for src in ['karate', 'chesapeake']:
    n0, E0 = read_gph(Q + src + '.gph'); N0 = [set() for _ in range(n0)]
    for a, b in E0: N0[a].add(b); N0[b].add(a)
    root = max(range(n0), key=lambda v: len(N0[v]))
    bfs = [root]; i = 0
    while len(bfs) < n0:
        for u in sorted(N0[bfs[i]]):
            if u not in bfs: bfs.append(u)
        i += 1
    for n in range(6, n0 + 1, 4):
        sel = bfs[:n]; idx = {v: k for k, v in enumerate(sel)}
        E = [(idx[a], idx[b]) for a, b in E0 if a in idx and b in idx]
        N = [set() for _ in range(n)]
        for a, b in E: N[a].add(b); N[b].add(a)
        order = degeneracy_order(N, n)
        circs = {'NF preparation A_q': prep_circuit(n, N, order, 0.9),
                 'CG-QAOA layer (p=1)': cg_qaoa(n, N, order, 0.9, [0.4], [0.6], measure=False),
                 'Penalty QAOA layer (p=1)': penalty_qaoa(n, E, 2.0, [0.4], [0.6], measure=False),
                 'Hadfield QAOA layer (p=1)': hadfield_qaoa(n, N, [0.4], [0.6], measure=False),
                 'NF preparation A_q (ancillas)': prep_circuit_anc(n, N, order, 0.9),
                 'CG-QAOA layer (ancillas)': cg_qaoa_anc(n, N, order, 0.9, [0.4], [0.6], measure=False)}
        for m, qc in circs.items():
            t0 = time.time()
            rec = dict(source=src, n=n, m=len(E), method=m, logical=counts(qc, False))
            if m != 'CG-QAOA layer (p=1)' or n <= 30: rec['routed'] = counts(qc, True)
            out['mis'].append(rec); print(src, n, m, rec['logical'], rec.get('routed'), f'{time.time()-t0:.0f}s', flush=True)
        json.dump(out, open('results/depth_scaling.json', 'w'), indent=1)
rng = random.Random(4)
for n in range(8, 41, 4):
    f = sample_satisfiable('3-SAT', n, rng, cap=20000); o = list(range(1, n + 1)); rng.shuffle(o)
    b = NegForcedSearch(f, o); b.prep(); A = b.qc.copy(); A.remove_final_measurements()
    b2 = NegForcedSearch(f, o); b2.iterate(); Q2 = b2.qc.copy(); Q2.remove_final_measurements()
    for m, qc in (('NF preparation A (3-SAT)', A), ('Grover iterate Q (3-SAT)', Q2)):
        rec = dict(n=n, m=f.m, method=m, logical=counts(qc, False)); out['sat'].append(rec); print('3-SAT', n, m, rec['logical'], flush=True)
    json.dump(out, open('results/depth_scaling.json', 'w'), indent=1)
