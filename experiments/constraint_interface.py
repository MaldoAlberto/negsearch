"""Random overlapping constraints (n=14, 8 constraints 'sum over 5 variables <= 2'), as in constraint_scaling.py.  Interface conditioning:
fix s variables classically (greedy: most frequent in the constraints), enumerate the feasible assignments of those s variables (the classical branches),
and compile the generic automaton circuit of the remaining n-s variables for a sample of branches.  Reports branches, mean/max width, mean compiled CZ/depth.
Exact enumeration + compiled counts; no noise, no device.  Writes results/constraint_interface.json."""
import os, sys, json
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); os.chdir(ROOT); sys.path.insert(0, ROOT)
import numpy as np
from qiskit import transpile
from negsearch.automaton import build_automaton, compile_Aq
N, SUB, B, M = 14, 5, 2, int(os.environ.get('CI_M', 8)); NS = int(os.environ.get('CI_SAMPLE', 6))
X = ((np.arange(2 ** N)[:, None] >> (N - 1 - np.arange(N))) & 1).astype(np.int8)
def cnt(qc):
    t = transpile(qc, basis_gates=['cx', 'rz', 'sx', 'x'], optimization_level=1, seed_transpiler=1)
    return int(t.count_ops().get('cx', 0)), int(t.depth())
res = {}
for seed in (1, 2):
    rng = np.random.default_rng(seed)
    cons = [(sorted(rng.choice(N, SUB, replace=False).tolist()), B) for _ in range(M)]
    ok = np.ones(2 ** N, bool)
    for S, b in cons: ok &= X[:, S].sum(1) <= b
    freq = np.zeros(N, int)
    for S, _ in cons:
        for i in S: freq[i] += 1
    order = list(np.argsort(-freq, kind='stable'))
    rows = []
    for s in (0, 1, 2, 3, 4, 6):
        I = sorted(order[:s]); R = [i for i in range(N) if i not in I]
        branches = {}
        for idx in np.flatnonzero(ok):
            key = tuple(X[idx, I]); branches.setdefault(key, []).append(idx)
        widths, comp = [], []
        keys = list(branches)
        for key in keys:
            sub = np.zeros(2 ** len(R), bool)
            for idx in branches[key]:
                v = 0
                for i in R: v = (v << 1) | int(X[idx, i])
                sub[v] = True
            A = build_automaton(sub, len(R)); widths.append(A.w)
        pick = rng.choice(len(keys), min(NS, len(keys)), replace=False)
        for pi in pick:
            key = keys[pi]; sub = np.zeros(2 ** len(R), bool)
            for idx in branches[key]:
                v = 0
                for i in R: v = (v << 1) | int(X[idx, i])
                sub[v] = True
            A = build_automaton(sub, len(R)); qc, _ = compile_Aq(A, 0.5); comp.append(cnt(qc))
        r = dict(s=s, free=len(R), branches=len(keys), width_mean=float(np.mean(widths)), width_max=int(max(widths)),
                 cz_mean=float(np.mean([c[0] for c in comp])), depth_mean=float(np.mean([c[1] for c in comp])))
        rows.append(r); print(seed, r, flush=True)
    res[str(seed)] = rows
json.dump(res, open('results/constraint_interface.json', 'w'))
