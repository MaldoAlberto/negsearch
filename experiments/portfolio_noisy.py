"""Noisy simulation (ibm_fez noise model, FakeFez) of the optimised circuits on QOBLIB po_a003_t02_orig:
does the feasibility guarantee survive noise, and how does the hybrid compare with penalty QAOA?
Parameters are the noiseless optima stored in results/portfolio_qaoa.json.
Usage: python experiments/portfolio_noisy.py [shots]"""
import os as _os, sys as _sys
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..'))
_os.chdir(ROOT); _sys.path.insert(0, ROOT)
import json, time, numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeFez
from negsearch.portfolio_circuits import qaoa_circuit, prep_circuit
from negsearch.portfolio_qaoa import Sim
from experiments.portfolio_qaoa import load
from experiments.portfolio_analysis import merged

KEY = 'a003_t02'
sig = lambda z: 1 / (1 + np.exp(-z))


def stats(P, S, counts):
    tot = sum(counts.values()); fe = qual = po = clean = 0
    for k, v in counts.items():
        b = np.array([int(ch) for ch in k.replace(' ', '')[::-1]])
        xb = b[:P.nq]
        ok = bool(P.feasible(xb[None])[0]); e = float(P.energy(xb[None])[0])
        if ok:
            fe += v; qual += v * (S.fworst - e) / (S.fworst - S.fopt); po += v * (e == S.fopt)
    return dict(p_feas=fe / tot, quality=qual / tot, p_opt=po / tot,
                quality_feasible_only=qual / max(fe, 1))   # after discarding infeasible samples classically


if __name__ == '__main__':
    SHOTS = int(_sys.argv[1]) if len(_sys.argv) > 1 else 2000
    P = load(KEY); S = Sim(P); R = merged()[KEY]
    sc = 1.0 / np.abs(S.f).max(); Delta = float(np.abs(P.h).max())
    bk = FakeFez(); noisy = AerSimulator.from_backend(bk)
    out = {}
    jobs = []
    for p in (1, 2):
        res = R[f'p{p}']
        a = res['penalty_unbalanced']['alpha']; hu, Ju, cu = P.unbalanced_poly(a * Delta, a * Delta)
        x = res['penalty_unbalanced']['x']
        jobs.append((f'penalty_unbalanced p={p}', qaoa_circuit(P, 'penalty_unbal', (x[:p], x[p:2 * p]),
                                                              h=hu * sc, J={k: v * sc for k, v in Ju.items()})))
        if 'penalty_slack' in res:
            a = res['penalty_slack']['alpha']; hs, Js, cs_, nt = P.slack_qubo(a * Delta)
            x = res['penalty_slack']['x']
            jobs.append((f'penalty_slack p={p}', qaoa_circuit(P, 'penalty_slack', (x[:p], x[p:2 * p]), h=hs * sc,
                                                             J={k: v * sc for k, v in Js.items()}, nq_total=nt)))
        x = res['A_init_XYmixer']['x']
        jobs.append((f'A_init_XYmixer p={p}', qaoa_circuit(P, 'cg_xy', (x[:p], x[p:2 * p]), q=sig(x[-1]),
                                                          h=P.h * sc, J={k: v * sc for k, v in P.J.items()})))
    jobs.append(('A_q alone', prep_circuit(P, R['A_alone']['q'])))
    for name, qc in jobs:
        qc = qc.copy(); qc.measure_all()
        t = transpile(qc, bk, optimization_level=3, seed_transpiler=1)
        t0 = time.time()
        cnt = noisy.run(t, shots=SHOTS, seed_simulator=1).result().get_counts()
        ideal = AerSimulator().run(transpile(qc, AerSimulator(), optimization_level=1), shots=SHOTS,
                                   seed_simulator=1).result().get_counts()
        out[name] = dict(cz=int(t.count_ops().get('cz', 0)), depth=int(t.depth()), shots=SHOTS,
                         ideal=stats(P, S, ideal), noisy=stats(P, S, cnt), seconds=time.time() - t0)
        print(name, json.dumps(out[name]), flush=True)
        json.dump(out, open('results/portfolio_noisy.json', 'w'), indent=1)
