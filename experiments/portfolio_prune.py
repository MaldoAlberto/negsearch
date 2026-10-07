"""Cheaper hybrid layers: (i) XY on a path instead of a ring inside each block, (ii) drop small objective couplings
from the phase operator (the sampled bitstrings are still scored with the exact objective).  Constraints live in
A_q, so the cost Hamiltonian has no penalty couplings and is cheap to sparsify.
Usage: python experiments/portfolio_prune.py"""
import os as _os, sys as _sys
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..'))
_os.chdir(ROOT); _sys.path.insert(0, ROOT)
import json, time, numpy as np
from qiskit import transpile
from qiskit_ibm_runtime.fake_provider import FakeFez
from negsearch.portfolio_circuits import qaoa_circuit
from negsearch.portfolio_qaoa import Sim
from experiments.portfolio_qaoa import INST, load, opt

sig = lambda z: 1 / (1 + np.exp(-z)); BK = FakeFez()
FRACS = (0.0, 0.01, 0.03, 0.1, 1.01)       # keep couplings with |J| >= frac * max|h|  (1.01 = linear phase only)

if __name__ == '__main__':
    fn = 'results/portfolio_prune.json'
    out = json.load(open(fn)) if _os.path.exists(fn) else {}
    for key in INST:
        P = load(key, order='shorts_first'); S = Sim(P); sc = 1 / np.abs(S.f).max(); hmax = float(np.abs(P.h).max())
        rec = out.get(key, {})
        qs = np.linspace(0.05, 0.95, 19); qa = qs[int(np.argmax([S.metrics(P.prep_state(q))['quality'] for q in qs]))]
        zq = np.log(qa / (1 - qa))
        for ring in (True, False):
            P.xy_ring = ring
            if hasattr(S, '_xy'): del S._xy
            for frac in FRACS:
                tag = f"{'ring' if ring else 'path'}_J{frac}"
                if tag in rec: continue
                Jp = {k: v for k, v in P.J.items() if abs(v) >= frac * hmax}
                cost = S.poly(P.h, Jp, P.const) * sc
                r = {'kept_couplings': len(Jp), 'all_couplings': sum(1 for v in P.J.values() if v)}
                rng = np.random.default_rng(5); prev = None
                for p in ((1, 2) if P.nq <= 16 else (1,)):
                    f = lambda x: float((np.abs(S.state('xy', x[:p], x[p:2 * p], cost, P.prep_state(sig(x[-1])))) ** 2 * cost).sum())
                    st = [np.r_[rng.uniform(0, 1.5, 2 * p), 0.0] for _ in range(3)] + [np.r_[np.zeros(2 * p), zq]]
                    if prev is not None: st.append(np.r_[prev[:1], prev[0], prev[1:2], prev[1], prev[2]])
                    o = opt(f, st); prev = o.x
                    m = S.metrics(S.state('xy', o.x[:p], o.x[p:2 * p], cost, P.prep_state(sig(o.x[-1]))))
                    qc = qaoa_circuit(P, 'cg_xy', (o.x[:p], o.x[p:2 * p]), q=sig(o.x[-1]), h=P.h * sc,
                                      J={k: v * sc for k, v in Jp.items()}, counter='lean')
                    qc.measure_all()
                    t = min((transpile(qc, BK, optimization_level=3, seed_transpiler=s) for s in (1, 2, 3, 4, 5)),
                            key=lambda c: c.count_ops().get('cz', 0))
                    r[f'p{p}'] = dict(x=list(map(float, o.x)), cz=int(t.count_ops().get('cz', 0)), depth=int(t.depth()), **m)
                rec[tag] = r
                print(key, tag, {k: ({kk: round(vv, 4) for kk, vv in v.items() if kk != 'x'} if isinstance(v, dict) else v)
                                 for k, v in r.items()}, flush=True)
                out[key] = rec; json.dump(out, open(fn, 'w'), indent=1)
