"""Objective layer only: default routing against a line swap network (couplings fused with swaps, Qiskit commuting-2q router) on the residual cores.  Writes results/swap_network_<tag>.json."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)
import json, os, warnings; warnings.filterwarnings('ignore')
import networkx as nx
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy, Commuting2qGateRouter, FindCommutingPauliEvolutions
import experiments.case_four as C
from negsearch.hw_large import ising_from_qubo
from negsearch.depth_study import phase_poly
from negsearch.fez_check import bk, esp_and_time

cp = C.core_problem(); free = cp['free']; col = cp['col']; f = len(free)
co = ising_from_qubo(cp['h1'], cp['J1'])
two = {(col[a], col[b]): c for (a, *r), c in [((k[0],) + k[1:], v) for k, v in co.items() if len(k) == 2] for b in r if a in col and b in col}
one = {col[k[0]]: v for k, v in co.items() if len(k) == 1 and k[0] in col}
print('free', f, 'couplings', len(two), 'dense would be', f * (f - 1) // 2, flush=True)
# default
qc = QuantumCircuit(f)
for q, c in one.items(): qc.rz(0.74 * c, q)
for (a, b), c in two.items(): qc.rzz(0.74 * c, a, b)
d = min((transpile(qc, bk, optimization_level=3, seed_transpiler=s) for s in range(1, 4)), key=lambda t: t.count_ops().get('cz', 0))
out = dict(free=f, couplings=len(two), default_cz=d.count_ops().get('cz', 0), default_depth=d.depth(), default_esp=esp_and_time(d)[0])
# swap network on a line of the device (needs ALL pairs of the line to be reachable: cost is that of the dense network)
H = SparsePauliOp.from_sparse_list([('ZZ', [a, b], c) for (a, b), c in two.items()], num_qubits=f)
q2 = QuantumCircuit(f)
for q, c in one.items(): q2.rz(0.74 * c, q)
q2.append(PauliEvolutionGate(H, 0.37), range(f))
G = nx.Graph(list(bk.coupling_map.get_edges()))
def dfs(path, seen):
    if len(path) == f: return path
    for u in sorted(G[path[-1]], key=lambda x: G.degree(x)):
        if u not in seen:
            r = dfs(path + [u], seen | {u})
            if r: return r
line = None
for s in sorted(G.nodes, key=lambda x: G.degree(x)):
    line = dfs([s], {s})
    if line: break
swap = SwapStrategy.from_line(list(range(f)))
pm = PassManager([FindCommutingPauliEvolutions(), Commuting2qGateRouter(swap, {(i, i + 1): i % 2 for i in range(f - 1)})])
t = transpile(pm.run(q2), bk, initial_layout=line, optimization_level=3, seed_transpiler=1, routing_method='none')
out.update(swapnet_cz=t.count_ops().get('cz', 0), swapnet_depth=t.depth(), swapnet_esp=esp_and_time(t)[0])
print(out, flush=True)
json.dump(out, open(f"results/swap_network_{os.environ.get('CF_TAG', 'n64')}.json", 'w'), indent=1)
