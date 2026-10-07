"""Head-to-head against other QUANTUM constraint handling: unbalanced penalisation and Fourier-based LCU (Carrera Vazquez, Egger, Woerner, arXiv:2605.18985).
Part 'quality' (exact, n=16 portfolio, same instance and objective as large_case_quality.py): feasibility and quality for
   ours (A_q + Grover mixer; conditioned + Grover mixer), penalty, unbalanced penalisation, and the single-basis-circuit variant of Fourier LCU
   (the count-type constraints are replaced by one layer of single-qubit Rz(theta_c a_ci), theta_c trained jointly with the QAOA angles; the
   exclusivity pairs stay coherent; X mixer; uniform start, or a trained global product-state bias = warm start).
Part 'compile' (n=8,16,24): CZ, depth and ESP on the ibm_fez topology of one layer p=1 of each method.
Writes results/vs_baselines.json."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)
import json, math, warnings; warnings.filterwarnings('ignore')
import numpy as np
from scipy.optimize import minimize


def lin_forms(A, T, sectors, cap=1, Dmax=3, Bmax=3):
    """constraints g >= 0 as (const, {var: coef}); variables (t,i,d) -> (t*A+i)*2+d"""
    Q = int(round(0.6 * T * Bmax)); idx = lambda t, i, d: (t * A + i) * 2 + d
    off = np.cumsum([0] + list(sectors)); forms = []
    for t in range(T):
        Ld = {idx(t, i, 0): 1.0 for i in range(A)}; Sd = {idx(t, i, 1): 1.0 for i in range(A)}
        diff = {**Ld, **{k: -1.0 for k in Sd}}; tot = {**Ld, **Sd}
        forms += [(0.0, diff), (float(Dmax), {k: -v for k, v in diff.items()}), (float(Bmax), {k: -1.0 for k in tot})]
        for s, size in enumerate(sectors):
            forms.append((float(cap), {idx(t, off[s] + j, 0): -1.0 for j in range(size)}))
    forms.append((float(Q), {idx(t, i, d): -1.0 for t in range(T) for i in range(A) for d in (0, 1)}))
    return forms


def penalty_qubo(A, T, sectors, w=1.0, l1=0.96, l2=0.037):
    nv = 2 * A * T; h = np.zeros(nv); J = {}
    for c0, a in lin_forms(A, T, sectors):
        # (-l1 g + l2 g^2) / 10 * w with g = c0 + sum a x
        sc = w / 10
        for v, ca in a.items():
            h[v] += sc * (-l1 * ca + l2 * 2 * c0 * ca)
        ks = list(a)
        for i, u in enumerate(ks):
            h[u] += sc * l2 * a[u] ** 2
            for v in ks[i + 1:]:
                key = (min(u, v), max(u, v)); J[key] = J.get(key, 0.0) + sc * l2 * 2 * a[u] * a[v]
    excl = {}
    for t in range(T):
        for i in range(A):
            key = ((t * A + i) * 2, (t * A + i) * 2 + 1); excl[key] = excl.get(key, 0.0) + w
    return h, J, excl, [a for _, a in lin_forms(A, T, sectors)]


def compile_part():
    from qiskit import QuantumCircuit
    from negsearch.hw_large import qubo_cost, ising_from_qubo, prune
    from negsearch.depth_study import phase_poly
    from experiments.qaoa_realistic import best as compile_best
    rows = []
    for A, T, S in [(4, 1, [2, 2]), (4, 2, [2, 2]), (6, 2, [3, 3])]:
        n = 2 * A * T
        h, J = qubo_cost(A, T, 5); obj = ising_from_qubo(h, J)
        hp, Jp, excl, forms = penalty_qubo(A, T, S)
        for pr in (0, 0.25):
            ob = prune(obj, pr) if pr else obj
            def circ(co):
                qc = QuantumCircuit(n); qc.h(range(n)); phase_poly(qc, co, 0.3)
                for q in range(n): qc.rx(0.8, q)
                return qc
            # unbalanced: objective + penalty (dense inside every constraint scope) + exclusivity
            ub = dict(ob); hh = hp.copy(); JJ = dict(Jp)
            for k, v in excl.items(): JJ[k] = JJ.get(k, 0) + v
            pen = ising_from_qubo(hh, JJ)
            for k, v in pen.items(): ub[k] = ub.get(k, 0) + v
            # Fourier LCU, one basis circuit: objective + coherent exclusivity + ONE Rz per qubit (all count constraints merged)
            lc = dict(ob)
            ex = ising_from_qubo(np.zeros(n), excl)
            for k, v in ex.items(): lc[k] = lc.get(k, 0) + v
            for q in range(n):
                th = sum(0.3 * a.get(q, 0.0) for a in forms); lc[(q,)] = lc.get((q,), 0) + th
            for name, co in (('unbalanced', ub), ('Fourier LCU (1 basis circuit)', lc)):
                r = compile_best(circ(co), seeds=4)
                r.update(n=n, method=name, pruned=bool(pr), zz=len([1 for k in co if len(k) == 2])); rows.append(r)
                print(f"n={n:3d} obj pruned={bool(pr)!s:5s} {name:30s} ZZ terms {r['zz']:4d}  CZ {r['cz']:5d} depth {r['depth']:5d} ESP {r['esp']:.3f}", flush=True)
    json.dump(rows, open('results/vs_baselines_compile.json', 'w'), indent=1)


def quality_part(seeds=(1, 2), depths=(1, 2, 3), restarts=4):
    import experiments.large_case_quality as LQ
    n = LQ.n; X = LQ.X; FEAS = LQ.FEAS; EXCL = LQ.EXCL; G = np.array(LQ.G, float)          # G: (C, 2^n) slack vectors (count constraints)
    pop = X.sum(1)
    out = {}
    for seed in seeds:
        c = LQ.make_cost(seed); cmin, cmax = c[FEAS].min(), c[FEAS].max(); cn = (c - cmin) / (cmax - cmin)
        opt = FEAS & (np.abs(c - cmin) < 1e-12)
        def metrics(P):
            f = float(P[FEAS].sum()); q = float((P * FEAS * (1 - cn)).sum())
            return dict(feas=f, popt=float(P[opt].sum()), quality=q, quality_feasible=q / f if f > 0 else 0.0)
        def state(par, p, w, warm):
            gam, bet, th = par[:p], par[p:2 * p], par[2 * p:2 * p + len(G)]
            if warm:
                ph = par[-1]; psi = (np.cos(ph / 2) ** (n - pop)) * (np.sin(ph / 2) ** pop)
            else:
                psi = np.ones(2 ** n) / math.sqrt(2 ** n)
            psi = psi.astype(complex)
            lin = np.exp(1j * (th @ G))
            for l in range(p):
                psi = psi * np.exp(-1j * gam[l] * (cn + w * EXCL)) * lin
                cc, ss = math.cos(bet[l]), -1j * math.sin(bet[l])
                t = psi.reshape([2] * n)
                for ax in range(n):
                    a0 = np.take(t, 0, axis=ax); a1 = np.take(t, 1, axis=ax)
                    t = np.stack([cc * a0 + ss * a1, ss * a0 + cc * a1], axis=ax)
                psi = t.reshape(-1)
            return psi
        res = {}
        for p in depths:
            for warm in (False, True):
                best = None
                for w in (1.0, 4.0):
                    rng = np.random.default_rng(seed * 100 + p)
                    for r in range(restarts):
                        x0 = np.concatenate([rng.uniform(-1, 1, p), rng.uniform(0, 1.5, p), rng.uniform(-1, 1, len(G))] + ([[rng.uniform(0.3, 1.2)]] if warm else []))
                        f = lambda x: -metrics(np.abs(state(x, p, w, warm)) ** 2)['quality']
                        o = minimize(f, x0, method='L-BFGS-B', options=dict(maxiter=60))
                        m = metrics(np.abs(state(o.x, p, w, warm)) ** 2)
                        if best is None or m['quality'] > best['quality']: best = m
                res[f"LCU{'+warm' if warm else ''} p={p}"] = best
                print(f"seed {seed} Fourier LCU (1 basis circuit){' + warm start' if warm else '':14s} p={p}: feas {best['feas']:.3f}  quality|feasible {best['quality_feasible']:.2f}  P(opt) {best['popt']:.4f}", flush=True)
        out[seed] = res
    json.dump(out, open('results/vs_baselines_quality.json', 'w'), indent=1)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'quality':
        quality_part()
    else:
        compile_part()
