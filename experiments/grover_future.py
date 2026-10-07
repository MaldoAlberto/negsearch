"""Fault-tolerant (future) cost of amplitude amplification over the coherent state A_q for the multi-constraint case.
Per iteration: A_q^dagger + reflection about |0..0> + A_q (the objective-threshold oracle is common to all methods and is excluded).
Gate model: a multi-controlled gate with c controls = AND-ladder of 2(c-1) Toffolis (c >= 2), Toffoli = 7 T gates (4 with relative phase).
Success probability a of the target set: iterations k = pi/(4 sqrt(a)); classical sampling from the same state needs 1/a samples.
Break-even a* from  k * T_iter * t_T = (1/a) * t_s.  Writes results/grover_future.json."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)
import json, math, warnings; warnings.filterwarnings('ignore')
import numpy as np
from negsearch.automaton import compile_Aq
from negsearch.large_case import case_automaton

CASES = [(6, 2, [3, 3]), (8, 3, [4, 4]), (10, 4, [5, 5]), (12, 8, [4, 4, 4])]
A_LEVELS = [1e-4, 1e-6, 1e-9, 1e-12]


def toffolis(qc):
    tof = 0
    for inst in qc.data:
        nc = getattr(inst.operation, 'num_ctrl_qubits', 0)
        if nc >= 2:
            tof += 2 * (nc - 1)
    return tof


def decisions(Aut):
    """expected number of decisions along the walk, and the maximum over feasible strings"""
    dist = np.zeros(Aut.width[0]); dist[0] = 1.0; ed = 0.0
    mx = [0] * 1
    best = [np.zeros(Aut.width[Aut.n])]
    for k in range(Aut.n - 1, -1, -1):
        bk = np.zeros(Aut.width[k])
        for s in range(Aut.width[k]):
            t0, t1 = Aut.trans[k][s]
            both = t0 >= 0 and t1 >= 0
            bk[s] = (1 if both else 0) + max(best[0][t] for t in (t0, t1) if t >= 0)
        best.insert(0, bk)
    for k in range(Aut.n):
        nd = np.zeros(Aut.width[k + 1])
        for s in range(Aut.width[k]):
            t0, t1 = Aut.trans[k][s]
            if t0 >= 0 and t1 >= 0:
                nd[t0] += dist[s] / 2; nd[t1] += dist[s] / 2; ed += dist[s]
            else:
                nd[t0 if t0 >= 0 else t1] += dist[s]
        dist = nd
    return ed, float(best[0][0])


def main():
    out = []
    for A, T, S in CASES:
        Aut, var = case_automaton(A, T, S, 1, 3, 3, int(round(0.6 * T * 3)), 'period')
        qc, info = compile_Aq(Aut, 0.5)
        N = qc.num_qubits
        tA = toffolis(qc); refl = 2 * (N - 1)
        tof_iter = 2 * tA + refl
        ed, dmax = decisions(Aut)
        row = dict(A=A, T=T, n=Aut.n, qubits=N, width=Aut.w, toffoli_A=tA, toffoli_iter=tof_iter, T_iter=7 * tof_iter, T_iter_rel=4 * tof_iter,
                   E_decisions=ed, D_max=dmax, a_lower_bound_opt=2.0 ** (-dmax), levels=[])
        for a in A_LEVELS:
            k = math.pi / (4 * math.sqrt(a))
            row['levels'].append(dict(a=a, iterations=k, T_total=k * 7 * tof_iter, t_quantum_s_opt=k * 7 * tof_iter * 1e-6, t_quantum_s_cons=k * 7 * tof_iter * 1e-5,
                                      samples=1 / a, t_classical_s_fast=1 / a * 1e-6, t_classical_s_slow=1 / a * 1e-5))
        row['a_breakeven_opt'] = (4 * 1e-5 / (math.pi * row['T_iter'] * 1e-6)) ** 2        # t_s = 10 us, t_T = 1 us
        row['a_breakeven_cons'] = (4 * 1e-6 / (math.pi * row['T_iter'] * 1e-5)) ** 2        # t_s = 1 us,  t_T = 10 us
        out.append(row)
        print(f"A={A},T={T} n={Aut.n}: N={N} qubits, Toffoli/iter {tof_iter:,}, T/iter {7*tof_iter:,.0f}; E[decisions] {ed:.1f}, D_max {dmax:.0f} (a_opt >= 2^-{dmax:.0f}); "
              f"break-even a* = {row['a_breakeven_opt']:.1e} (optimistic) / {row['a_breakeven_cons']:.1e} (conservative)", flush=True)
        for L in row['levels']:
            print(f"     a={L['a']:.0e}: k={L['iterations']:.2e} iterations, {L['T_total']:.2e} T gates, quantum {L['t_quantum_s_opt']:.1e}-{L['t_quantum_s_cons']:.1e} s, classical sampling {L['t_classical_s_fast']:.1e}-{L['t_classical_s_slow']:.1e} s")
    json.dump(out, open('results/grover_future.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
