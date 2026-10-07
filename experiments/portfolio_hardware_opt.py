"""Can the hybrid (A_q + XY) beat unbalanced-penalty QAOA on today's hardware?
Cheaper preparation: variables ordered shorts-first (fewer forced states) and per-period counters that are never
uncomputed (valid for the XY hybrid).  Re-optimise the hybrid for that state (exact simulation), count CZ/depth on
ibm_fez (best of 5 transpiler seeds, same for every circuit), and run the ibm_fez noise model on po_a003_t02_orig.
Measured counters give a free consistency check (counter value == number of positions measured in that period).
Usage: python experiments/portfolio_hardware_opt.py [shots]"""
import os as _os, sys as _sys
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..'))
_os.chdir(ROOT); _sys.path.insert(0, ROOT)
import json, time, numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeFez
from negsearch.portfolio_circuits import qaoa_circuit, prep_circuit_lean
from negsearch.portfolio_qaoa import Sim
from experiments.portfolio_qaoa import INST, load, opt
from experiments.portfolio_analysis import merged

sig = lambda z: 1 / (1 + np.exp(-z))
BK = FakeFez()


def transp(qc, seeds=(1, 2, 3, 4, 5)):
    best = None
    for s in seeds:
        t = transpile(qc, BK, optimization_level=3, seed_transpiler=s)
        if best is None or t.count_ops().get('cz', 0) < best.count_ops().get('cz', 0): best = t
    return best


def stats(P, S, counts, wl=None, ws=None):
    tot = sum(counts.values()); fe = qual = po = 0; cons = cons_q = cons_po = 0
    for k, v in counts.items():
        b = np.array([int(ch) for ch in k.replace(' ', '')[::-1]])
        xb = b[:P.nq]
        ok = bool(P.feasible(xb[None])[0]); e = float(P.energy(xb[None])[0])
        g = (S.fworst - e) / (S.fworst - S.fopt) if ok else 0.0
        if ok: fe += v; qual += v * g; po += v * (e == S.fopt)
        if wl is not None:                       # counter consistency check
            L, Sc = P.counts(xb[None]); pos = P.nq; good = True
            for t in range(P.T):
                cl = sum(int(b[pos + j]) << j for j in range(wl)); pos += wl
                cs = sum(int(b[pos + j]) << j for j in range(ws)); pos += ws
                good &= (cl == L[0, t]) and (cs == Sc[0, t])
            if good: cons += v; cons_q += v * g; cons_po += v * (ok and e == S.fopt)
    r = dict(p_feas=fe / tot, quality=qual / tot, p_opt=po / tot, quality_feasible_only=qual / max(fe, 1))
    if wl is not None:
        r.update(kept_by_counter_check=cons / tot, quality_after_check=cons_q / max(cons, 1),
                 p_opt_after_check=cons_po / max(cons, 1))
    return r


if __name__ == '__main__':
    SHOTS = int(_sys.argv[1]) if len(_sys.argv) > 1 else 2000
    R = merged(); out = {}
    for key in INST:
        P = load(key, order='shorts_first'); S = Sim(P); sc = 1 / np.abs(S.f).max(); cost = S.f * sc
        D = float(np.abs(P.h).max()); rec = {}
        qs = np.linspace(0.05, 0.95, 19); mq = [S.metrics(P.prep_state(q)) for q in qs]
        b = int(np.argmax([m['quality'] for m in mq])); rec['A_alone'] = dict(q=float(qs[b]), **mq[b])
        zq = np.log(qs[b] / (1 - qs[b])); rng = np.random.default_rng(3); prev = None
        for p in (1, 2):
            f = lambda x: float((np.abs(S.state('xy', x[:p], x[p:2 * p], cost, P.prep_state(sig(x[-1])))) ** 2 * cost).sum())
            st = [np.r_[rng.uniform(0, 1.5, 2 * p), 0.0] for _ in range(3)] + [np.r_[np.zeros(2 * p), zq]]
            if prev is not None: st.append(np.r_[prev[:1], prev[0], prev[1:2], prev[1], prev[2]])
            r = opt(f, st); prev = r.x
            rec[f'hybrid_p{p}'] = dict(q=float(sig(r.x[-1])), x=list(map(float, r.x)),
                                       **S.metrics(S.state('xy', r.x[:p], r.x[p:2 * p], cost, P.prep_state(sig(r.x[-1])))))
        A = prep_circuit_lean(P, rec['A_alone']['q']); wl = len(A.qregs[1]); ws = len(A.qregs[2])
        circs = {'A_q alone (lean)': A}
        for p in (1, 2):
            x = np.array(rec[f'hybrid_p{p}']['x'])
            circs[f'hybrid p={p} (lean)'] = qaoa_circuit(P, 'cg_xy', (x[:p], x[p:2 * p]), q=sig(x[-1]), h=P.h * sc,
                                                         J={k: v * sc for k, v in P.J.items()}, counter='lean')
            u = R[key][f'p{p}']['penalty_unbalanced']; a = u['alpha']; hu, Ju, cu = P.unbalanced_poly(a * D, a * D)
            xu = np.array(u['x'])
            circs[f'unbalanced p={p}'] = qaoa_circuit(P, 'penalty_unbal', (xu[:p], xu[p:2 * p]), h=hu * sc,
                                                      J={k: v * sc for k, v in Ju.items()})
            rec[f'unbalanced_p{p}'] = {k: v for k, v in u.items() if k != 'x'}
        rec['hardware'] = {}
        for name, qc in circs.items():
            qm = qc.copy(); qm.measure_all(); t = transp(qm)
            h = dict(qubits=qc.num_qubits, cz=int(t.count_ops().get('cz', 0)), depth=int(t.depth()),
                     cz_depth=int(t.depth(lambda ins: ins.operation.num_qubits == 2)))
            if key == 'a003_t02':
                t0 = time.time()
                cnt = AerSimulator.from_backend(BK).run(t, shots=SHOTS, seed_simulator=1).result().get_counts()
                lean = 'lean' in name
                h['noisy'] = stats(P, S, cnt, wl if lean else None, ws if lean else None); h['seconds'] = time.time() - t0
            rec['hardware'][name] = h
            print(key, name, json.dumps(h), flush=True)
        out[key] = rec
        print(key, {k: {kk: round(vv, 4) for kk, vv in v.items() if kk in ('p_feas', 'quality', 'p_opt', 'q')}
                    for k, v in rec.items() if k != 'hardware'}, flush=True)
        json.dump(out, open('results/portfolio_hardware_opt.json', 'w'), indent=1)
