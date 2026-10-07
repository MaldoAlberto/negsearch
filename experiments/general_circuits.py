"""Generality test 3 (gate level, Qiskit Aer): the generic compiler automaton -> circuit, then
  (a) Grover: standard (H^n diffusion) vs amplitude amplification about A_q|0>, success probability vs iterations,
  (b) Grover-mixer QAOA p=2 circuit vs the exact numpy state,
on six constraint families with n = 6 decision variables.  Oracle and cost layer are exact diagonal gates (identical in the
two methods, hence not counted).  CZ counts: transpiled to {cz, rz, sx, x, ry}, optimisation level 1.
Writes results/general_circuits.json."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)
import json, math
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from negsearch.automaton import (build_automaton, compile_Aq, mu_q, prep_state, gm_qaoa_state, to_qiskit, from_qiskit,
                                 append_diag, reflect_zero, gm_mixer, best_k, grover_curve)
from negsearch import families as fam

SIM = AerSimulator(method='statevector')
BASIS = ['cz', 'rz', 'sx', 'x', 'ry']


def statevector(qc):
    c = qc.copy(); c.save_statevector()
    return np.asarray(SIM.run(transpile(c, SIM, optimization_level=0)).result().get_statevector().data)


def px_of(sv, n, nq):
    p = (np.abs(sv) ** 2).reshape(2 ** (nq - n), 2 ** n).sum(0)
    return from_qiskit(p, n)


def cz_count(qc):
    t = transpile(qc, basis_gates=BASIS, optimization_level=1)
    return t.count_ops().get('cz', 0), t.depth()


def grover_circuit(P, A, k, mode):
    n = P.n
    good = P.good(0.0)
    phases = to_qiskit(np.where(good, -1.0, 1.0).astype(complex), n)
    if mode == 'std':
        qc = QuantumCircuit(n)
        qc.h(range(n))
        for _ in range(k):
            append_diag(qc, list(range(n)), phases)
            qc.h(range(n)); reflect_zero(qc, list(range(n)), math.pi); qc.h(range(n))     # -(2|0><0|-I)... up to phase
        return qc, n
    Aq, info = compile_Aq(A, 0.5)
    nq = Aq.num_qubits
    qc = QuantumCircuit(nq)
    qc.compose(Aq, inplace=True)
    for _ in range(k):
        append_diag(qc, list(range(n)), phases)
        gm_mixer(qc, Aq, math.pi)
    return qc, nq


def main():
    probs = [fam.cardinality_qp(6, 3, 0), fam.knapsack(6, 11), fam.mis(6, 0.4, 22), fam.set_packing(6, 5, 33),
             fam.exact_cover(6, 4, 44), fam.colouring(3, 55, 0.7)]
    out = []
    rng = np.random.default_rng(0)
    for P in probs:
        if P.nF < 2:
            continue
        n = P.n
        A = build_automaton(P.feas, n)
        good = P.good(0.0)
        Aq, info = compile_Aq(A, 0.5)
        cz_A, d_A = cz_count(Aq)
        mu, _ = mu_q(A, 0.5)
        a = float(mu[good].sum()); a0 = good.sum() / 2 ** n
        ks = best_k(a); k0 = best_k(a0)
        # (a) Grover curves vs theory
        curve_std = grover_curve(np.ones(2 ** n) / math.sqrt(2 ** n), good, k0 + 1)[0]
        curve_aq = grover_curve(np.sqrt(mu), good, ks + 1)[0]
        meas_std, meas_aq = [], []
        for k in range(k0 + 2):
            qc, nq = grover_circuit(P, A, k, 'std')
            meas_std.append(float(px_of(statevector(qc), n, nq)[good].sum()))
        for k in range(ks + 2):
            qc, nq = grover_circuit(P, A, k, 'aq')
            meas_aq.append(float(px_of(statevector(qc), n, nq)[good].sum()))
        err = max(np.abs(np.array(meas_std) - curve_std[:len(meas_std)]).max(),
                  np.abs(np.array(meas_aq) - curve_aq[:len(meas_aq)]).max())
        # (b) GM-QAOA p=2 circuit vs numpy
        span = P.worst - P.opt
        c = (P.cost - P.opt) / span
        params = rng.uniform(0, 2 * math.pi, 4)
        qc = QuantumCircuit(Aq.num_qubits)
        qc.compose(Aq, inplace=True)
        for l in range(2):
            append_diag(qc, list(range(n)), to_qiskit(np.exp(-1j * params[l] * c), n))
            gm_mixer(qc, Aq, params[2 + l])
        sv = statevector(qc)
        p_circ = px_of(sv, n, qc.num_qubits)
        p_np = np.abs(gm_qaoa_state(np.sqrt(mu), c, params)) ** 2
        err_q = float(np.abs(p_circ - p_np).max())
        feas_mass = float(p_circ[P.feas].sum())
        row = dict(name=P.name, n=n, nq=Aq.num_qubits, nF=P.nF, nG=int(good.sum()), width=A.w, cz_Aq=cz_A, depth_Aq=d_A,
                   k_std=k0, k_aq=ks, p_std=meas_std[k0], p_aq=meas_aq[ks], a_unif=a0, a_aq=a, max_err_grover=float(err),
                   max_err_qaoa=err_q, qaoa_feas=feas_mass)
        out.append(row)
        print(f"{P.name[:36]:36s} qubits={Aq.num_qubits:2d} CZ(A_q)={cz_A:4d}  Grover k: {k0}->{ks}  P_succ {meas_std[k0]:.3f}->{meas_aq[ks]:.3f}"
              f"  |circuit-theory| grover={err:.1e} qaoa={err_q:.1e} P(feas)={feas_mass:.6f}", flush=True)
    json.dump(out, open('results/general_circuits.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
