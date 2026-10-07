"""Depth-reduction tools for constraint-guided circuits.
- log-depth AND tree for the n-qubit reflection (instead of a linear chain)
- dependency-layered vertex orders (graph-colouring layers) for the preparation
- swap-strategy routing for penalty QAOA (baseline)"""
import numpy as np
from qiskit import QuantumCircuit
from .qaoa_circuits import prep_circuit_anc

def _tree_and(qc, leaves, anc):
    """Compute AND(leaves) with a balanced tree of rccx; returns (top two nodes, list of ops)."""
    level, ops, k = list(leaves), [], 0
    while len(level) > 2:
        nxt = []
        for i in range(0, len(level) - 1, 2):
            a = anc[k]; k += 1; qc.rccx(level[i], level[i + 1], a); ops.append((level[i], level[i + 1], a)); nxt.append(a)
        if len(level) % 2: nxt.append(level[-1])
        level = nxt
    return level, ops

def mc_zero_phase_tree(qc, qubits, anc, phi):
    """Phase e^{i phi} on |0...0> of `qubits`, depth O(log n), using len(qubits)-2 ancillas."""
    qc.x(qubits)
    top, ops = _tree_and(qc, qubits, anc)
    if len(top) == 1: qc.p(phi, top[0])
    else: qc.cp(phi, top[0], top[1])
    for a, b, c in reversed(ops): qc.rccx(a, b, c)
    qc.x(qubits)

def cg_qaoa_tree(n, N, order, q, gammas, betas, measure=True):
    A0 = prep_circuit_anc(n, N, order, q)
    n_anc = max(A0.num_qubits - n, n - 2, 1)
    A = prep_circuit_anc(n, N, order, q, n_anc=n_anc); Ad = prep_circuit_anc(n, N, order, q, inverse=True, n_anc=n_anc)
    qc = QuantumCircuit(n + n_anc, n); anc = list(range(n, n + n_anc))
    qc.compose(A, inplace=True)
    for g, b in zip(gammas, betas):
        for v in range(n): qc.p(g, v)
        qc.compose(Ad, inplace=True)
        mc_zero_phase_tree(qc, list(range(n)), anc, -b)
        qc.compose(A, inplace=True)
    if measure: qc.measure(range(n), range(n))
    return qc

def coloring_layer_order(N, n):
    """Greedy (largest-first) colouring; process colour classes in turn so non-adjacent vertices share layers."""
    col = {}
    for v in sorted(range(n), key=lambda u: -len(N[u])):
        used = {col[u] for u in N[v] if u in col}; c = 0
        while c in used: c += 1
        col[v] = c
    return sorted(range(n), key=lambda v: (col[v], len(N[v]), v))

def degeneracy_order(N, n):
    rem = set(range(n)); seq = []
    while rem:
        v = min(rem, key=lambda u: (len(N[u] & rem), u)); seq.append(v); rem.remove(v)
    return seq[::-1]

def penalty_qaoa_swapstrategy(n, E, lam, gamma, beta, backend):
    """Penalty QAOA p=1 routed with a line swap strategy (Qiskit commuting-2q-gate router)."""
    import networkx as nx
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.transpiler import PassManager, CouplingMap
    from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy, Commuting2qGateRouter, FindCommutingPauliEvolutions
    from qiskit.transpiler.passes import SetLayout, FullAncillaAllocation, EnlargeWithAncilla, ApplyLayout
    from qiskit import transpile
    deg = [0] * n
    for a, b in E: deg[a] += 1; deg[b] += 1
    H = SparsePauliOp.from_sparse_list([('ZZ', [a, b], lam / 4) for a, b in E], num_qubits=n)
    qc = QuantumCircuit(n); qc.h(range(n))
    for v in range(n): qc.rz(2 * gamma * (0.5 - lam * deg[v] / 4), v)
    qc.append(PauliEvolutionGate(H, gamma), range(n)); qc.rx(2 * beta, range(n)); qc.measure_all()
    # find a line of n physical qubits on the device
    G = nx.Graph(list(backend.coupling_map.get_edges()))
    def dfs(path, seen):
        if len(path) == n: return path
        for u in sorted(G[path[-1]], key=lambda x: G.degree(x)):
            if u not in seen:
                r = dfs(path + [u], seen | {u})
                if r: return r
        return None
    line = None
    for s in sorted(G.nodes, key=lambda x: G.degree(x)):
        line = dfs([s], {s})
        if line: break
    swap = SwapStrategy.from_line(list(range(n)))
    edge_coloring = {(i, i + 1): i % 2 for i in range(n - 1)}
    pm = PassManager([FindCommutingPauliEvolutions(), Commuting2qGateRouter(swap, edge_coloring)])
    routed = pm.run(qc)
    return transpile(routed, backend, initial_layout=line, optimization_level=3, seed_transpiler=1, routing_method='none')
