"""Structured case: n fixed, m disjoint cardinality constraints (exactly g/2 of each group of g=n/m variables), plus one global budget over the groups.
Ours (structured): one Dicke block per group, run in parallel; the global budget is not in the circuit: it selects which tuples of group counts are enumerated classically (interface).
Penalty/unbalanced layer for the same constraints: one ZZ per pair inside a group.  Generic automaton circuit (n=12 only, for scale).
Compiled counts to {cx,rz,sx,x}, all-to-all; exact enumeration for the tuples.  Writes results/constraint_scaling_structured.json."""
import os, sys, json, itertools, math
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); os.chdir(ROOT); sys.path.insert(0, ROOT)
import numpy as np
from qiskit import QuantumCircuit, transpile
from negsearch.dicke import dicke_circuit
from negsearch.automaton import build_automaton, compile_Aq
LOG = ['cx', 'rz', 'sx', 'x']
def cnt(qc):
    t = transpile(qc, basis_gates=LOG, optimization_level=1, seed_transpiler=1)
    return dict(cz=int(t.count_ops().get('cx', 0)), depth=int(t.depth()))
out = dict(structured=[], generic=[])
N = 24
for m in [1, 2, 3, 4, 6, 8, 12]:
    g = N // m; k = g // 2
    d = cnt(dicke_circuit(g, k)) if g > 1 else dict(cz=0, depth=1)
    pen = QuantumCircuit(N)
    for j in range(m):
        for a, c in itertools.combinations(range(j * g, (j + 1) * g), 2): pen.cx(a, c); pen.rz(0.2, c); pen.cx(a, c)
    # global budget: sum of group 'levels' l_j in {0..g} with exactly k each  -> tuples of per-group choices; here: each group may take any count c_j in [0,g]; budget sum c_j = N/2
    # (so the per-group cardinality becomes a menu; count tuples)
    tuples = sum(1 for c in itertools.product(range(g + 1), repeat=m) if sum(c) == N // 2) if m <= 6 else None
    r = dict(m=m, g=g, k=k, ours=dict(cz=m * d['cz'], depth=d['depth'], qubits=N), block=d, penalty=cnt(pen), pairs=m * g * (g - 1) // 2, budget_tuples=tuples)
    out['structured'].append(r); print(r, flush=True)
n = 12
for m in [1, 2, 3, 4, 6]:
    g = n // m; k = g // 2
    X = ((np.arange(2 ** n)[:, None] >> (n - 1 - np.arange(n))) & 1)
    ok = np.ones(2 ** n, bool)
    for j in range(m): ok &= X[:, j * g:(j + 1) * g].sum(1) == k
    A = build_automaton(ok, n); qc, _ = compile_Aq(A, 0.5)
    t = cnt(qc); sym = cnt(dicke_circuit(g, k)) if g > 1 else dict(cz=0, depth=1)
    r = dict(m=m, g=g, F=int(ok.sum()), width=int(A.w), generic=dict(t, qubits=qc.num_qubits), structured=dict(cz=m * sym['cz'], depth=sym['depth']))
    out['generic'].append(r); print(r, flush=True)
json.dump(out, open('results/constraint_scaling_structured.json', 'w'))
