"""QOBLIB 06-portfolio: negation-forced state A_q as QAOA initial state / mixer vs penalty encodings.
Exact statevector simulation.  Usage: python experiments/portfolio_qaoa.py [PMAX] [instance-keys...]"""
import os as _os, sys as _sys
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..'))
_os.chdir(ROOT); _sys.path.insert(0, ROOT)
import numpy as np, json, time
from scipy.optimize import minimize
from negsearch.portfolio import Instance, Portfolio
from negsearch.portfolio_qaoa import Sim

DATA = 'instances/qoblib_portfolio'      # copies of the QOBLIB data (CC BY 4.0), see the README there
INST = {   # key: (instance directory, B, lambda, cs2)   [ub = 1, C = 3, cs1 = 2]
    'a003_t02': ('po_a003_t02_orig', 3, '1e-05', 2),
    'a004_t02': ('po_a004_t04_orig-T2', 4, '2e-05', 3),
    'a005_t02': ('po_a005_t04_orig-T2', 4, '4e-05', 3),
}
sig = lambda z: 1 / (1 + np.exp(-z))


def opt(f, x0s, maxiter=200):
    best = None
    for x0 in x0s:
        r = minimize(f, x0, method='COBYLA', options=dict(maxiter=maxiter, rhobeg=0.3))
        if best is None or r.fun < best.fun: best = r
    return best


def load(key, order='interleaved'):
    name, B, lam, cs2 = INST[key]
    return Portfolio(Instance(f'{DATA}/{name}'), B, lam, cs2=cs2, order=order)


if __name__ == '__main__':
    PMAX = int(_sys.argv[1]) if len(_sys.argv) > 1 else 3
    keys = _sys.argv[2:] or list(INST)
    out_fn = 'results/portfolio_qaoa.json'
    out = json.load(open(out_fn)) if _os.path.exists(out_fn) else {}
    for key in keys:
        P = load(key); S = Sim(P)
        sc = 1.0 / np.abs(S.f).max()                    # common energy scale for gamma
        cost = S.f * sc
        Delta = float(np.abs(P.h).max()) * sc            # largest single-variable coefficient (scaled)
        R = dict(qoblib=INST[key][0], periods=P.T, assets=P.na, B=P.B, C=P.C, lam=INST[key][2], ub=1,
                 qubits_x=P.nq, qubits_slack_qubo=P.nq + P.T * (P.cs1 + P.cs2),
                 feasible_per_period=P.feasible_count_per_period(), n_feasible=int(S.feas.sum()),
                 f_opt=float(S.fopt), f_worst_feasible=float(S.fworst),
                 opt_solution=P.solution_lines(S.bits[np.argmax(S.opt), :P.nq]))
        R['uniform'] = S.metrics(np.ones(2 ** S.n) / np.sqrt(2 ** S.n))
        # A_q alone (classically samplable): best q for quality
        qs = np.linspace(0.05, 0.95, 19)
        mq = [S.metrics(P.prep_state(q)) for q in qs]
        b = int(np.argmax([m['quality'] for m in mq]))
        R['A_alone'] = dict(q=float(qs[b]), **mq[b]); R['A_q0.5'] = S.metrics(P.prep_state(0.5))
        print(key, {k: R[k] for k in ('qubits_x', 'qubits_slack_qubo', 'n_feasible', 'f_opt')},
              'uniform', R['uniform'], 'A', R['A_alone'], flush=True)
        rng = np.random.default_rng(7); prev = {}
        for p in range(1, PMAX + 1):
            t0 = time.time(); res = {}

            def starts(k, dim, extra=()):
                s = [np.r_[rng.uniform(0, 1.5, p), rng.uniform(0, 1.5, p), list(extra)] for _ in range(3)]
                if k in prev:
                    x = prev[k]; pp = p - 1
                    s.append(np.r_[x[:pp], x[pp - 1], x[pp:2 * pp], x[2 * pp - 1], x[2 * pp:]])
                return s

            # ---- penalty QAOA, unbalanced penalisation (no slack), |+>, X mixer
            best = None
            for a in (0.5, 2.0, 8.0):
                l1 = l2 = a * Delta
                hu, Ju, cu = P.unbalanced_poly(l1 / sc, l2 / sc)
                cpen = S.poly(hu, Ju, cu) * sc
                f = lambda x: float((np.abs(S.state('x', x[:p], x[p:], cpen)) ** 2 * cpen).sum())
                r = opt(f, starts(f'unb{a}', 2 * p)); prev[f'unb{a}'] = r.x
                m = S.metrics(S.state('x', r.x[:p], r.x[p:], cpen))
                if best is None or m['quality'] > best['quality']: best = dict(alpha=a, **m)
            res['penalty_unbalanced'] = best
            # ---- penalty QAOA with the reference slack QUBO (only if it fits in memory)
            nsl = P.T * (P.cs1 + P.cs2)
            if P.nq + nsl <= 20:
                Ssl = Sim(P, extra=nsl); best = None
                for a in (0.5, 2.0, 8.0):
                    hs, Js, cs_, ntot = P.slack_qubo(a * Delta / sc)
                    cpen = Ssl.poly(hs, Js, cs_) * sc
                    f = lambda x: float((np.abs(Ssl.state('x', x[:p], x[p:], cpen)) ** 2 * cpen).sum())
                    r = opt(f, starts(f'sl{a}', 2 * p)); prev[f'sl{a}'] = r.x
                    m = Ssl.metrics(Ssl.state('x', r.x[:p], r.x[p:], cpen))
                    if best is None or m['quality'] > best['quality']: best = dict(alpha=a, **m)
                res['penalty_slack'] = best; del Ssl
            # ---- A_q as initial state only: unbalanced cost, X mixer (q variational)
            a_w = res['penalty_unbalanced']['alpha']
            hu, Ju, cu = P.unbalanced_poly(a_w * Delta / sc, a_w * Delta / sc)
            cpen = S.poly(hu, Ju, cu) * sc
            f = lambda x: float((np.abs(S.state('x', x[:p], x[p:2 * p], cpen, P.prep_state(sig(x[-1])))) ** 2 * cpen).sum())
            zq = np.log(R['A_alone']['q'] / (1 - R['A_alone']['q']))   # also start at gamma=beta=0, q of A_alone
            r = opt(f, starts('warm', 2 * p, (0.0,)) + [np.r_[np.zeros(2 * p), zq]]); prev['warm'] = r.x
            res['A_init_Xmixer'] = dict(q=float(sig(r.x[-1])), alpha=a_w,
                                        **S.metrics(S.state('x', r.x[:p], r.x[p:2 * p], cpen, P.prep_state(sig(r.x[-1])))))
            # ---- A_q + XY mixer inside (period, direction) blocks: feasibility preserved
            f = lambda x: float((np.abs(S.state('xy', x[:p], x[p:2 * p], cost, P.prep_state(sig(x[-1])))) ** 2 * cost).sum())
            r = opt(f, starts('xy', 2 * p, (0.0,)) + [np.r_[np.zeros(2 * p), zq]]); prev['xy'] = r.x
            res['A_init_XYmixer'] = dict(q=float(sig(r.x[-1])),
                                         **S.metrics(S.state('xy', r.x[:p], r.x[p:2 * p], cost, P.prep_state(sig(r.x[-1])))))
            # ---- CG-QAOA: A_q + Grover mixer about A_q|0>
            f = lambda x: float((np.abs(S.state('grover', x[:p], x[p:2 * p], cost, P.prep_state(sig(x[-1])))) ** 2 * cost).sum())
            r = opt(f, starts('gm', 2 * p, (0.0,)) + [np.r_[np.zeros(2 * p), zq]]); prev['gm'] = r.x
            res['A_init_Grover_mixer'] = dict(q=float(sig(r.x[-1])),
                                              **S.metrics(S.state('grover', r.x[:p], r.x[p:2 * p], cost, P.prep_state(sig(r.x[-1])))))
            res['params'] = {k: list(map(float, v)) for k, v in prev.items()}
            R[f'p{p}'] = res
            print(f' {key} p={p} ({time.time() - t0:.0f}s)',
                  {k: {kk: round(vv, 4) for kk, vv in v.items() if kk in ('p_feas', 'p_opt', 'quality', 'q', 'alpha')}
                   for k, v in res.items() if k != 'params'}, flush=True)
            out[key] = R
            json.dump(out, open(out_fn, 'w'), indent=1)
