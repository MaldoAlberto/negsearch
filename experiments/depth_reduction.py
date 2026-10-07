"""Depth-reduction study on MIS (BFS subgraphs of QOBLIB karate): orders, log-depth reflection, swap strategy."""
import os as _os, sys as _sys
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT); _sys.path.insert(0, ROOT)
import json, numpy as np
from negsearch.qaoa_mis import read_gph, MIS, prep_biased
from negsearch.qaoa_circuits import prep_circuit, prep_circuit_anc, cg_qaoa_anc, penalty_qaoa
from negsearch.depth_tools import cg_qaoa_tree, coloring_layer_order, degeneracy_order, penalty_qaoa_swapstrategy
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime.fake_provider import FakeFez
fake = FakeFez(); LOG = ['cx', 'rz', 'sx', 'x']
def cost(qc, routed):
    qc = qc.copy(); qc.remove_final_measurements()
    best = None
    for s in (range(3) if routed else [0]):
        t = transpile(qc, fake, optimization_level=3, seed_transpiler=s) if routed else transpile(qc, basis_gates=LOG, optimization_level=3)
        r = (t.depth(), sum(v for k, v in t.count_ops().items() if k in ('cx', 'cz')))
        if best is None or r[0] < best[0]: best = r
    return dict(depth=best[0], twoq=best[1])
# --- correctness of the tree reflection on a small graph ---
rng = np.random.default_rng(5); n = 7; E = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.45]
g = MIS(n, E); o = degeneracy_order(g.N, n); q = 0.8
psi0 = prep_biased(g, o, q); psi = psi0 * np.exp(-1j * 0.4 * (-g.size)); psi = psi - (1 - np.exp(-1j * 1.1)) * psi0 * np.vdot(psi0, psi)
sv = Statevector(cg_qaoa_tree(n, g.N, o, q, [0.4], [1.1], measure=False)).data[:2 ** n]
print('tree-reflection CG-QAOA == numpy:', abs(abs(np.vdot(sv, psi)) - 1) < 1e-9, flush=True)
# --- study ---
Q = _os.environ.get('QOBLIB_MIS', 'qoblib/07-independentset/') + 'instances/'
n0, E0 = read_gph(Q + 'karate.gph'); N0 = [set() for _ in range(n0)]
for a, b in E0: N0[a].add(b); N0[b].add(a)
root = max(range(n0), key=lambda v: len(N0[v])); bfs = [root]; i = 0
while len(bfs) < n0:
    for u in sorted(N0[bfs[i]]):
        if u not in bfs: bfs.append(u)
    i += 1
out = []
for n in (14, 18, 26, 34):
    idx = {v: k for k, v in enumerate(bfs[:n])}; E = [(idx[a], idx[b]) for a, b in E0 if a in idx and b in idx]
    N = [set() for _ in range(n)]
    for a, b in E: N[a].add(b); N[b].add(a)
    orders = {'low-degree first': sorted(range(n), key=lambda v: (len(N[v]), v)), 'degeneracy': degeneracy_order(N, n), 'colouring layers': coloring_layer_order(N, n)}
    rec = dict(n=n, m=len(E))
    for on, o in orders.items():
        pos = {v: k for k, v in enumerate(o)}
        rec[f'A[{on}]'] = dict(max_controls=max(len([u for u in N[v] if pos[u] < pos[v]]) for v in range(n)),
                               logical=cost(prep_circuit_anc(n, N, o, 0.9), False), routed=cost(prep_circuit_anc(n, N, o, 0.9), True))
    o = orders['degeneracy']
    rec['CG layer chain'] = dict(logical=cost(cg_qaoa_anc(n, N, o, 0.9, [0.4], [0.6]), False), routed=cost(cg_qaoa_anc(n, N, o, 0.9, [0.4], [0.6]), True))
    rec['CG layer tree'] = dict(logical=cost(cg_qaoa_tree(n, N, o, 0.9, [0.4], [0.6]), False), routed=cost(cg_qaoa_tree(n, N, o, 0.9, [0.4], [0.6]), True))
    rec['penalty default'] = dict(routed=cost(penalty_qaoa(n, E, 2.0, [0.4], [0.6]), True))
    t = penalty_qaoa_swapstrategy(n, E, 2.0, 0.4, 0.6, fake)
    rec['penalty swap strategy'] = dict(routed=dict(depth=t.depth(), twoq=t.count_ops().get('cz', 0)))
    out.append(rec); print(json.dumps(rec), flush=True)
json.dump(out, open('results/depth_reduction.json', 'w'), indent=1)
