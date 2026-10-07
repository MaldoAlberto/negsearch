"""Generality test 5: the generic compiler at scale.  Constraint automata are written from the problem STRUCTURE (no truth table),
compiled to circuits (one exact unitary per multi-controlled operation, registers interleaved with the variables), and simulated
with Qiskit Aer's matrix-product-state backend.  Reported per instance: qubits, automaton width, MPS bond dimension, share of
feasible samples (should be 100 %), mean objective of the samples against the exact value under A_q (dynamic programme over the
automaton), the exact optimum (DP) and the best of the shots, and the share of feasible strings among all 2^n (the uniform state).
Writes results/general_scaling_mps.json."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)
import json, time, re
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from negsearch.structural import (knapsack_automaton, cardinality_automaton, banded_is_automaton, count_feasible, marginals,
                                  linear_opt)
from negsearch.automaton import compile_Aq

SHOTS = 2000
rng0 = np.random.default_rng(7)


def instance(kind, n):
    rng = np.random.default_rng(1000 * n + len(kind))
    if kind == 'knapsack':
        w = rng.integers(1, 4, n); v = rng.integers(2, 20, n); W = int(0.4 * w.sum())
        return knapsack_automaton(list(map(int, w)), W), v.astype(float), 'max', dict(W=W)
    if kind == 'cardinality':
        K = max(2, n // 6); v = rng.uniform(1, 10, n)
        return cardinality_automaton(n, K), v, 'max', dict(K=K)
    if kind == 'independent set (band d=2)':
        v = rng.integers(1, 9, n).astype(float)
        return banded_is_automaton(n, 2), v, 'max', dict(d=2)


def run(kind, n):
    A, v, sense, meta = instance(kind, n)
    t = time.time()
    qc, info = compile_Aq(A, 0.5, interleave=True, native=True)
    xs = info['x']
    c = QuantumCircuit(qc.num_qubits, n); c.compose(qc, inplace=True)
    for i, q in enumerate(xs):
        c.measure(q, i)
    sim = AerSimulator(method='matrix_product_state')
    r = sim.run(c, shots=SHOTS, seed_simulator=1).result()
    counts = r.get_counts()
    tsim = time.time() - t
    # bond dimension of the final state
    c2 = qc.copy(); c2.save_matrix_product_state()
    r2 = AerSimulator(method='matrix_product_state').run(c2).result()
    mps = r2.data(0)['matrix_product_state']
    bond = max([g[0].shape[1] if hasattr(g[0], 'shape') else 1 for g in mps[0]] + [1])
    # decode: classical bit i of the count key is x_i (little endian in the key string)
    vals, feas = [], 0
    wts = v
    nq_per = None
    cnt_tot = 0
    for key, m in counts.items():
        bits = [int(ch) for ch in key[::-1]]          # bits[i] = x_i
        cnt_tot += m
        vals += [float(np.dot(bits, wts))] * m
    # feasibility check through the automaton: replay
    def feasible(bits):
        s = 0
        for k, b in enumerate(bits):
            s = A.trans[k][s][b]
            if s < 0:
                return False
        return True
    feas = sum(m for key, m in counts.items() if feasible([int(ch) for ch in key[::-1]]))
    ex = float(np.dot(marginals(A, 0.5), v))
    opt, _ = linear_opt(A, v, sense)
    nf = count_feasible(A)
    return dict(kind=kind, n=n, qubits=qc.num_qubits, width=A.w, bond=int(bond), shots=SHOTS, p_feas=feas / SHOTS,
                p_uniform_feas=float(nf / 2 ** n), mean_sample=float(np.mean(vals)), mean_exact=ex,
                se=float(np.std(vals) / np.sqrt(SHOTS)), best=float(max(vals)), opt=float(opt), seconds=time.time() - t, **meta)


def main(sizes=None):
    sizes = sizes or {'cardinality': [12, 24, 36, 48, 60], 'independent set (band d=2)': [12, 24, 36, 48, 60, 80],
                      'knapsack': [12, 18, 24, 30]}
    rows = []
    for kind, ns in sizes.items():
        for n in ns:
            try:
                row = run(kind, n)
            except Exception as e:                      # keep going; record the failure
                row = dict(kind=kind, n=n, error=repr(e)[:200])
            rows.append(row)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
            json.dump(rows, open('results/general_scaling_mps.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
