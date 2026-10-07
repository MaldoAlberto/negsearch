"""Does the feasibility block grow with the number of constraints?  Fixed n, constraints added one by one.
Ours: generic automaton circuit A_q (best of NP variable orders by max width); counts compiled to {cx,rz,sx,x}, all-to-all (logical).
Penalty / unbalanced constraint layer: the diagonal phase of sum_j [-l1 g_j + l2 g_j^2], g_j = sum_{i in S_j} x_i - b_j  (no slack): one ZZ per distinct pair, one Rz per variable.
Families: 'random' (random 5-subsets, sum <= 2), 'groups' (m disjoint exactly-n/(2m)... cardinality groups, structured).
Writes results/constraint_scaling.json.  Exact enumeration of F; compiled counts only (no noise, no device)."""
import os, sys, json, itertools, math, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); os.chdir(ROOT); sys.path.insert(0, ROOT)
import numpy as np
from qiskit import QuantumCircuit, transpile
from negsearch.automaton import build_automaton, compile_Aq
LOG = ['cx', 'rz', 'sx', 'x']
N = int(os.environ.get('CS_N', 14)); SEEDS = [int(s) for s in os.environ.get('CS_SEEDS', '1,2,3').split(',')]
NP = int(os.environ.get('CS_NP', 8)); M = int(os.environ.get('CS_M', 12)); SUB = int(os.environ.get('CS_SUB', 5)); B = int(os.environ.get('CS_B', 2))
X = ((np.arange(2 ** N)[:, None] >> (N - 1 - np.arange(N))) & 1).astype(np.int8)       # row index = MSB-first string (variable 0 = most significant)

def cnt(qc):
    t = transpile(qc, basis_gates=LOG, optimization_level=1, seed_transpiler=1)
    return dict(cz=int(t.count_ops().get('cx', 0)), depth=int(t.depth()), qubits=int(qc.num_qubits))

def penalty_layer(cons):
    """cons: list of (S, b).  Diagonal phase circuit of the sum of -l1 g + l2 g^2 over constraints (angles arbitrary: counts do not depend on them)."""
    pairs, single = set(), set()
    for S, b in cons:
        single |= set(S)
        pairs |= {(a, c) for a, c in itertools.combinations(sorted(S), 2)}
    qc = QuantumCircuit(N)
    for i in sorted(single): qc.rz(0.37, i)
    for a, c in sorted(pairs):
        qc.cx(a, c); qc.rz(0.21, c); qc.cx(a, c)
    return qc, len(pairs)

def feas_of(cons, perm):
    ok = np.ones(2 ** N, bool)
    for S, b in cons:
        ok &= X[:, [perm[i] for i in S]].sum(1) <= b if b >= 0 else True
    return ok

def run_family(fam, seed):
    rng = np.random.default_rng(seed)
    if fam == 'random':
        cons_all = [(sorted(rng.choice(N, SUB, replace=False).tolist()), B) for _ in range(M)]
        seq = [cons_all[:m] for m in range(0, M + 1)]
    else:                                   # disjoint groups of size g, 'at most g/2'; m groups -> n = m*g only if divisible: use g = N//m for m dividing N... instead nest by pairs
        raise SystemExit
    out = []
    for m, cons in enumerate(seq):
        best = None
        perms = [list(range(N))] + [rng.permutation(N).tolist() for _ in range(NP - 1)]
        for perm in perms:                  # perm relabels variables: position i of the circuit holds variable perm[i]
            ok = feas_of(cons, perm)
            A = build_automaton(ok, N)
            if best is None or A.w < best[0].w: best = (A, perm, ok)
        A, perm, ok = best
        qc, info = compile_Aq(A, 0.5)
        r = dict(m=m, F=int(ok.sum()), width=int(A.w), ours=cnt(qc))
        pl, npairs = penalty_layer([([perm.index(i) for i in S], b) for S, b in cons])
        r['penalty'] = cnt(pl) if m else dict(cz=0, depth=0, qubits=N); r['pairs'] = npairs
        out.append(r); print(fam, seed, m, r, flush=True)
    return out

if __name__ == '__main__':
    res = {'n': N, 'sub': SUB, 'b': B, 'random': {str(s): run_family('random', s) for s in SEEDS}}
    json.dump(res, open('results/constraint_scaling.json', 'w'))
