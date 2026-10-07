"""3-SAT: depth per round and total depth to reach a solution (rounds x depth), Grover vs negation-forced (NF)."""
import os as _os, sys as _sys
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT); _sys.path.insert(0, ROOT)
import json, math, random, statistics
from negsearch.core import sample_satisfiable, enumerate_models, success_by_solutions
from negsearch.circuits import NegForcedSearch
from qiskit import transpile
LOG = ['cx', 'rz', 'sx', 'x']
def depth(qc):
    qc = qc.copy(); qc.remove_final_measurements(); t = transpile(qc, basis_gates=LOG, optimization_level=1)
    return t.depth(), t.count_ops().get('cx', 0)
rng = random.Random(4); out = []
for n in range(8, 41, 4):
    rows = []
    for inst in range(3):
        f = sample_satisfiable('3-SAT', n, rng, cap=20000); sols = enumerate_models(f)
        o = list(range(1, n + 1)); rng.shuffle(o); pN, _ = success_by_solutions(f, o, sols); pG = len(sols) / 2 ** n
        a = NegForcedSearch(f, o); a.prep(); dA, cA = depth(a.qc)
        q = NegForcedSearch(f, o); q.iterate(); dQ, cQ = depth(q.qc)
        g = NegForcedSearch(f, o, baseline=True); g.iterate(); dG, cG = depth(g.qc)
        kN, kG = math.pi / (4 * math.sqrt(pN)), math.pi / (4 * math.sqrt(pG))
        rows.append(dict(n=n, pN=pN, pG=pG, depth_A=dA, depth_round_NF=dQ, depth_round_Grover=dG, cx_round_NF=cQ, cx_round_Grover=cG,
                         rounds_NF=kN, rounds_Grover=kG, total_depth_NF=dA + kN * dQ, total_depth_Grover=kG * dG))
    med = {k: statistics.median(r[k] for r in rows) for k in rows[0]}
    out.append(med); print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in med.items()}, flush=True)
json.dump(out, open('results/depth_to_solution_3sat.json', 'w'), indent=1)
