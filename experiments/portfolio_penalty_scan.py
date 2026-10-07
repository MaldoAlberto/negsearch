"""Fairness check for the penalty baselines: the main run used penalty weights alpha in {0.5, 2, 8} x max|h| and
alpha = 0.5 (the grid edge) always won.  Re-optimise with smaller weights (default alpha in {0.0625, 0.125, 0.25}; env PEN_ALPHAS, PEN_OUT).
Usage: python experiments/portfolio_penalty_scan.py [PMAX] [keys...]"""
import os as _os, sys as _sys
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..'))
_os.chdir(ROOT); _sys.path.insert(0, ROOT)
import numpy as np, json, time
from negsearch.portfolio_qaoa import Sim
from experiments.portfolio_qaoa import INST, load, opt

ALPHAS = tuple(float(a) for a in _os.environ.get('PEN_ALPHAS', '0.0625,0.125,0.25').split(','))
if __name__ == '__main__':
    PMAX = int(_sys.argv[1]) if len(_sys.argv) > 1 else 3
    keys = _sys.argv[2:] or list(INST)
    fn = _os.environ.get('PEN_OUT', 'results/portfolio_penalty_scan.json')
    out = json.load(open(fn)) if _os.path.exists(fn) else {}
    for key in keys:
        P = load(key); S = Sim(P); sc = 1.0 / np.abs(S.f).max(); D = float(np.abs(P.h).max())
        nsl = P.T * (P.cs1 + P.cs2); Ssl = Sim(P, extra=nsl) if P.nq + nsl <= 20 else None
        rng = np.random.default_rng(11); prev = {}; R = out.get(key, {})
        for p in range(1, PMAX + 1):
            t0 = time.time(); res = {}
            def starts(k):
                s = [np.r_[rng.uniform(0, 1.5, p), rng.uniform(0, 1.5, p)] for _ in range(3)]
                if k in prev:
                    x = prev[k]; pp = p - 1; s.append(np.r_[x[:pp], x[pp - 1], x[pp:2 * pp], x[2 * pp - 1]])
                return s
            for name, SS in (('penalty_unbalanced', S), ('penalty_slack', Ssl)):
                if SS is None: continue
                best = None
                for a in ALPHAS:
                    if name == 'penalty_unbalanced':
                        hh, JJ, cc = P.unbalanced_poly(a * D, a * D)
                    else:
                        hh, JJ, cc, _ = P.slack_qubo(a * D)
                    cpen = SS.poly(hh, JJ, cc) * sc
                    f = lambda x: float((np.abs(SS.state('x', x[:p], x[p:], cpen)) ** 2 * cpen).sum())
                    r = opt(f, starts(f'{name}{a}')); prev[f'{name}{a}'] = r.x
                    m = SS.metrics(SS.state('x', r.x[:p], r.x[p:], cpen))
                    if best is None or m['quality'] > best['quality']: best = dict(alpha=a, **m)
                res[name] = best
            res['params'] = {k: list(map(float, v)) for k, v in prev.items()}
            R[f'p{p}'] = res
            print(key, f'p={p} ({time.time()-t0:.0f}s)', {k: v for k, v in res.items() if k != 'params'}, flush=True)
            out[key] = R; json.dump(out, open(fn, 'w'), indent=1)
