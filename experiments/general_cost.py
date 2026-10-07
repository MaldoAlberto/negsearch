"""Cost accounting per amplification iteration (CZ after transpilation to {cz, rz, sx, x, ry}, no hand optimisation):
  standard Grover : feasibility oracle (compute automaton, phase, uncompute)  +  diffusion H^n (MCZ_n) H^n
  amplification on A_q : A_q^dagger, A_q, and the phase flip on |0...0> of all qubits (decisions + registers)
The objective-threshold oracle is identical in both and is not counted.  Totals = iterations x per-iteration cost.
Writes results/general_cost.json."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)
import json, math
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCPhaseGate, MCXGate
from negsearch.automaton import build_automaton, compile_Aq, compile_feas_oracle, mu_q, best_k
from negsearch import families as fam

BASIS = ['cz', 'rz', 'sx', 'x', 'ry']


def cz(qc):
    t = transpile(qc, basis_gates=BASIS, optimization_level=1)
    return t.count_ops().get('cz', 0)


def mcz_cost(N):
    qc = QuantumCircuit(N)
    if N == 1:
        qc.z(0)
    else:
        qc.append(MCPhaseGate(math.pi, N - 1), range(N))
    return cz(qc)


def mcz_vchain(N):
    """phase flip on |1...1> of N qubits using N-2 clean ancillas (V-chain of Toffolis)"""
    if N <= 3:
        return mcz_cost(N)
    from qiskit.circuit.library import MCXVChain
    g = MCXVChain(N - 1, dirty_ancillas=False)
    qc = QuantumCircuit(g.num_qubits)
    qc.h(N - 1); qc.append(g, range(g.num_qubits)); qc.h(N - 1)
    return cz(qc)


def account(P):
    A = build_automaton(P.feas, P.n)
    n = P.n
    good = P.good(0.0)
    mu, _ = mu_q(A, 0.5)
    k_std = best_k(good.sum() / 2 ** n); k_aq = best_k(float(mu[good].sum()))
    Aq, _ = compile_Aq(A, 0.5)
    orc, _ = compile_feas_oracle(A)
    c_Aq = cz(Aq); c_or = cz(orc)
    c_diff = mcz_cost(n); c_zero = mcz_cost(Aq.num_qubits)
    it_std = c_or + c_diff
    it_aq = 2 * c_Aq + c_zero
    c_diff_v = mcz_vchain(n); c_zero_v = mcz_vchain(Aq.num_qubits)
    it_std_v = c_or + c_diff_v; it_aq_v = 2 * c_Aq + c_zero_v
    return dict(name=P.name, n=n, qubits_aq=Aq.num_qubits, qubits_std=orc.num_qubits, width=A.w, cz_Aq=c_Aq, cz_oracle=c_or,
                cz_diffusion=c_diff, cz_zero_reflection=c_zero, iter_std=it_std, iter_aq=it_aq, k_std=k_std, k_aq=k_aq,
                total_std=k_std * it_std, total_aq=c_Aq + k_aq * it_aq,
                cz_diffusion_v=c_diff_v, cz_zero_v=c_zero_v, iter_std_v=it_std_v, iter_aq_v=it_aq_v,
                total_std_v=k_std * it_std_v, total_aq_v=c_Aq + k_aq * it_aq_v)


def main():
    ps = [fam.cardinality_qp(6, 3, 0), fam.knapsack(6, 11), fam.mis(6, 0.4, 22), fam.set_packing(6, 5, 33),
          fam.exact_cover(6, 4, 44), fam.colouring(3, 55, 0.7),
          fam.cardinality_qp(10, 4, 0), fam.knapsack(10, 10), fam.mis(10, 0.35, 20), fam.set_packing(10, 7, 30),
          fam.exact_cover(10, 6, 40), fam.colouring(5, 50), fam.assignment_qap(3, 60), fam.assignment_qap(4, 7)]
    rows = []
    for P in ps:
        if P.nF < 2:
            continue
        r = account(P)
        rows.append(r)
        print(f"{P.name[:34]:34s} n={r['n']:2d} w={r['width']:2d} A_q={r['cz_Aq']:5d} oracle={r['cz_oracle']:5d} diff={r['cz_diffusion']:4d} "
              f"zero={r['cz_zero_reflection']:6d} | iter std={r['iter_std']:6d} aq={r['iter_aq']:6d} | k {r['k_std']}->{r['k_aq']} "
              f"| total std={r['total_std']:7d} aq={r['total_aq']:7d} | vchain: zero={r['cz_zero_v']} std={r['total_std_v']} aq={r['total_aq_v']}", flush=True)
        json.dump(rows, open('results/general_cost.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
