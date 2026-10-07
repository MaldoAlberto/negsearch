"""Architecture study: one 10-qubit block of the 120-qubit portfolio, compiled to different connectivities and two-qubit gate sets.
Compiled counts only (no noise simulation).  ESP uses a uniform error model (e2 per two-qubit gate, e1 per one-qubit gate, readout er);
default e2=0.0035 is the effective CZ error measured on ibm_fez compilations in this paper.  Nothing is executed on a device."""
import json, math, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from qiskit import transpile
from qiskit.transpiler import CouplingMap
from negsearch import hw100 as H

NQ = H.NB_Q
def topologies():
    t = {}
    t['all-to-all'] = CouplingMap.from_full(NQ)
    t['line'] = CouplingMap.from_line(NQ, bidirectional=True)
    t['ring'] = CouplingMap.from_ring(NQ, bidirectional=True)
    t['grid 2x5'] = CouplingMap.from_grid(2, 5, bidirectional=True)
    t['grid 3x4 (10 of 12)'] = CouplingMap.from_grid(3, 4, bidirectional=True)
    t['heavy-hex (12-ring patch)'] = CouplingMap.from_heavy_hex(3, bidirectional=True)
    return t

def run(methods, block=1, e2=0.0035, e1=0.00025, seeds=6, out='results/architectures.json'):
    inst = H.build_instance()
    b = inst['blocks'][block]; inst1 = dict(inst); inst1['blocks'] = [b]
    par = H.train_all(inst1, log=lambda *a: None)[0]
    rows = []
    gatesets = {'CZ': ['cz', 'rz', 'sx', 'x'], 'CZ+RZZ(fractional)': ['cz', 'rzz', 'rz', 'sx', 'x']}
    for m in methods:
        qc = H.block_circuit(b, m, par)
        for tn, cm in topologies().items():
            if tn.startswith('heavy-hex'):
                # keep the first 12 qubits' connected subgraph large enough: use heavy-hex of distance 3 and take a connected 12-qubit region
                full = cm; nodes = list(range(full.size()))
                import networkx as nx
                g = nx.Graph([tuple(e) for e in full.get_edges()])
                comp = list(nx.bfs_tree(g, 0).nodes())[:NQ]
                sub = g.subgraph(comp); mp = {n: k for k, n in enumerate(comp)}
                cm = CouplingMap([[mp[a], mp[c]] for a, c in sub.edges()] + [[mp[c], mp[a]] for a, c in sub.edges()])
            for gn, bg in gatesets.items():
                best = None
                for sd in range(seeds):
                    t = transpile(qc, coupling_map=cm, basis_gates=bg, optimization_level=3, seed_transpiler=sd + 1)
                    ops = t.count_ops(); n2 = ops.get('cz', 0) + ops.get('rzz', 0)
                    key = (n2, t.depth(lambda i: len(i.qubits) == 2))
                    if best is None or key < best[0]: best = (key, t)
                t = best[1]; ops = t.count_ops(); n2 = ops.get('cz', 0) + ops.get('rzz', 0)
                n1 = sum(v for k, v in ops.items() if k in ('rz', 'sx', 'x'))
                esp = math.exp(n2 * math.log1p(-e2) + n1 * math.log1p(-e1))
                rows.append(dict(block=block, ls=list(b['ls']), F=int(b['mask'].sum()), method=m, topology=tn, gates=gn, two_q=n2, depth=t.depth(), depth_2q=t.depth(lambda i: len(i.qubits) == 2), esp=esp))
                print(rows[-1], flush=True)
    json.dump(rows, open(out, 'w'), indent=1)
    return rows

if __name__ == '__main__':
    blk = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    run(['ours_p0', 'ours', 'lcu', 'unbalanced', 'xy'], block=blk, out=f'results/architectures_b{blk}.json')
