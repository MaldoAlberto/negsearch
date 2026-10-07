"""Mid-circuit measurement / classical randomness in the preparation block of the hardware-friendly hybrid
(shorts-first order, XY on a path, linear phase).  Feasibility depends only on the counts (L, S) of each period and
the XY mixer and the phase never change those counts, so different count sectors never interfere.  Hence:
  * measuring the counters right after A_q leaves every statistic of x unchanged (= coherent A_q);
  * the sector can be drawn classically (offline, or with a mid-circuit-measured coin) and only the superposition
    inside the sector has to be prepared on the chip: a Dicke state per (period, direction) block, or simply one
    basis state;
  * measuring the decision qubits themselves (measure-and-flip) turns A_q into a classical mixture of basis states.
All are simulated exactly; sector weights are those of A_q (one coin bias q, optimised with the angles).
Usage: python experiments/portfolio_midmeasure.py"""
import os as _os, sys as _sys
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..'))
_os.chdir(ROOT); _sys.path.insert(0, ROOT)
import json, itertools, numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime.fake_provider import FakeFez
from negsearch.portfolio_qaoa import Sim
from negsearch.portfolio_circuits import xy_edges, cost_layer, prep_circuit_lean, qaoa_circuit
from negsearch.dicke import dicke
from qiskit.circuit.library import XXPlusYYGate
from experiments.portfolio_qaoa import load, opt

sig = lambda z: 1 / (1 + np.exp(-z)); BK = FakeFez()


def sectors(P, S):
    """sector id of every basis state: tuple of (L_t, S_t) -> integer label (feasible states only)."""
    L, Sc = P.counts(S.bits[:, :P.nq])
    key = np.zeros(len(L), np.int64)
    for t in range(P.T): key = key * 64 + L[:, t] * 8 + Sc[:, t]
    return key


def init_states(P, S, q, kind, sec):
    a = P.prep_state(q)
    if kind == 'A_q coherent': return a
    p = np.abs(a) ** 2; out = np.zeros_like(a)
    for s in np.unique(sec[S.feas]):
        m = (sec == s) & S.feas; w = p[m].sum()
        if kind == 'sector + Dicke': out[m] = np.sqrt(w / m.sum())
        elif kind == 'sector + basis state': out[np.nonzero(m)[0][0]] = np.sqrt(w)
    return out


def mixed_measured(S, P, q, gam, bet, cost):
    """A_q fully measured (classical mixture of feasible basis states with weights |A_q(x)|^2)."""
    a = P.prep_state(q); idx = np.nonzero(np.abs(a) > 1e-12)[0]
    M = np.zeros((2 ** S.n, len(idx)), complex); M[idx, np.arange(len(idx))] = 1
    for g, b in zip(gam, bet):
        M = M * np.exp(-1j * g * cost)[:, None]; M = S.xy_mixer(M, b)
    return (np.abs(M) ** 2 * (np.abs(a[idx]) ** 2)[None, :]).sum(1)


def metrics_p(S, pr):
    qual = (S.fworst - S.f) / (S.fworst - S.fopt)
    return dict(p_feas=float(pr[S.feas].sum()), p_opt=float(pr[S.opt].sum()), quality=float((pr * S.feas * qual).sum()))


def cz(qc):
    qc = qc.copy(); qc.measure_all()
    t = min((transpile(qc, BK, optimization_level=3, seed_transpiler=s) for s in (1, 2, 3, 4, 5)),
            key=lambda c: c.count_ops().get('cz', 0))
    return int(t.count_ops().get('cz', 0)), int(t.depth())


def sector_circuit(P, counts, gam, bet, h, prep):
    """counts[t] = (L_t, S_t); prep in {'dicke', 'basis'}; XY on a path, linear phase."""
    qc = QuantumCircuit(P.nq)
    for t, (L, Sc) in enumerate(counts):
        for d, k in ((0, L), (1, Sc)):
            blk = [P.q(t, d, i) for i in range(P.na)]
            if prep == 'dicke': dicke(qc, blk, k)
            else:
                for i in range(k): qc.x(blk[i])
    for g, b in zip(gam, bet):
        cost_layer(qc, list(range(P.nq)), h, {}, g)
        for Lr in xy_edges(P):
            for a_, b_ in Lr: qc.append(XXPlusYYGate(2 * b), [a_, b_])
    return qc


if __name__ == '__main__':
    out = {}
    for key in ('a003_t02', 'a004_t02'):
        P = load(key, order='shorts_first'); P.xy_ring = False; S = Sim(P); sec = sectors(P, S)
        sc = 1 / np.abs(S.f).max(); cost = S.poly(P.h, {}, P.const) * sc
        rec = {}
        for p in (1, 2):
            for kind in ('A_q coherent', 'sector + Dicke', 'sector + basis state') + (('A_q measured (classical)',) if key == 'a003_t02' else ()):
                if kind == 'A_q measured (classical)':
                    f = lambda x: float((mixed_measured(S, P, sig(x[-1]), x[:p], x[p:2 * p], cost) * cost).sum())
                else:
                    f = lambda x: float((np.abs(S.state('xy', x[:p], x[p:2 * p], cost, init_states(P, S, sig(x[-1]), kind, sec))) ** 2 * cost).sum())
                rng = np.random.default_rng(2)
                st = [np.r_[rng.uniform(0, 1.5, 2 * p), z] for z in (-2.0, -1.0, 0.0)] + [np.r_[np.zeros(2 * p), -2.5]]
                r = opt(f, st, maxiter=250); x = r.x
                if kind == 'A_q measured (classical)':
                    pr = mixed_measured(S, P, sig(x[-1]), x[:p], x[p:2 * p], cost)
                else:
                    pr = np.abs(S.state('xy', x[:p], x[p:2 * p], cost, init_states(P, S, sig(x[-1]), kind, sec))) ** 2
                m = dict(q=float(sig(x[-1])), x=list(map(float, x)), **metrics_p(S, pr))
                # hardware cost (worst sector for Dicke; any sector for basis; lean A_q for coherent)
                if kind == 'A_q coherent':
                    m['cz'], m['depth'] = cz(qaoa_circuit(P, 'cg_xy', (x[:p], x[p:2 * p]), q=sig(x[-1]), h=P.h * sc, J={}, counter='lean'))
                elif kind in ('sector + Dicke', 'sector + basis state'):
                    secs = [s for s in itertools.product(*[[(L, Sc) for L in range(P.na + 1) for Sc in range(P.na + 1) if P.ok_counts(L, Sc)]] * P.T)]
                    worst = max(secs, key=lambda c: sum(min(L, P.na - L) + min(Sc, P.na - Sc) for L, Sc in c))
                    m['cz'], m['depth'] = cz(sector_circuit(P, worst, x[:p], x[p:2 * p], P.h * sc, 'dicke' if 'Dicke' in kind else 'basis'))
                    m['worst_sector'] = worst
                rec[f'p{p} | {kind}'] = m
                print(key, p, kind, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if k != 'x'}, flush=True)
        out[key] = rec
        json.dump(out, open('results/portfolio_midmeasure.json', 'w'), indent=1)
