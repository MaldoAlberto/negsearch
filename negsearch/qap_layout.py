"""Initial layout as a quadratic assignment problem: place logical qubits on physical qubits minimising
sum_ij flow(i,j) * (dist(p_i,p_j) - 1) (+ optional error weight), solved by simulated annealing; then routed by Sabre with that initial layout."""
import numpy as np
import rustworkx as rx
from qiskit import transpile


def flow_matrix(qc):
    lg = transpile(qc, basis_gates=['cz', 'rz', 'sx', 'x', 'ry'], optimization_level=3, seed_transpiler=1)
    N = qc.num_qubits; F = np.zeros((N, N))
    for inst in lg.data:
        if inst.operation.name == 'cz':
            a, b = (lg.find_bit(q).index for q in inst.qubits); F[a, b] += 1; F[b, a] += 1
    return F


def qap_layout(F, backend, iters=60000, seed=0, err_weight=0.0):
    cm = backend.coupling_map; g = cm.graph.to_undirected(multigraph=False)
    P = g.num_nodes(); D = np.array(rx.distance_matrix(g)) ; N = len(F)
    err = np.zeros(P)
    if err_weight:
        for q in range(P):
            e = backend.target['measure'][(q,)].error or 0; err[q] = e
    rng = np.random.default_rng(seed)
    adj = [list(g.neighbors(i)) for i in range(P)]
    # start: a connected blob grown from a random node
    best = None
    for r in range(8):
        start = int(rng.integers(P)); blob = [start]
        while len(blob) < N:
            cand = [v for u in blob for v in adj[u] if v not in blob]; blob.append(int(rng.choice(cand)))
        pos = np.array(blob); rng.shuffle(pos)
        cost = lambda pos: float((F * np.maximum(D[np.ix_(pos, pos)] - 1, 0)).sum() / 2) + err_weight * err[pos].sum()
        c = cost(pos); T = max(c, 1.0) * 0.3
        for it in range(iters // 8):
            T *= 0.9998
            new = pos.copy()
            if rng.random() < 0.5:
                i, j = rng.choice(N, 2, replace=False); new[i], new[j] = new[j], new[i]
            else:
                i = int(rng.integers(N)); nb = [v for v in adj[new[i]] if v not in new]
                if not nb:
                    continue
                new[i] = int(rng.choice(nb))
            cn = cost(new)
            if cn < c or rng.random() < np.exp((c - cn) / max(T, 1e-9)):
                pos, c = new, cn
        if best is None or c < best[0]:
            best = (c, pos.copy())
    return best[1].tolist(), best[0]
