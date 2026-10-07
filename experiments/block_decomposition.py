"""QAOA at >100 variables as many small block circuits (separator decomposition).
Given an interface tuple (per-period, per-sector (l,s) counts), the problem splits into blocks (period t, sector s) of 2*size variables.
Block coordinate scheme: every block is optimised by a p=1 QAOA (pair-swap mixers, feasibility preserved) while all other variables are fixed;
the fixed neighbours enter as linear fields.  Compile-only for the circuits (ibm_fez topology), exact simulation for the quality.
Writes results/block_decomposition.json (blocks prepared with the symmetry-aware lowering of A_q)."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)
import json, itertools, warnings, time; warnings.filterwarnings('ignore')
import numpy as np
from scipy.optimize import minimize
from qiskit import transpile
from negsearch.hw_large import (qubo_cost, ising_from_qubo, build_hw_circuit_swap, block_strings, tensor_qaoa_swap, x_index, prune)
from negsearch.large_case import sector_block_automaton
from negsearch.fez_check import bk, esp_and_time
from negsearch.dicke import dicke
from negsearch.hw_large import swap_mixer_layer
from negsearch.depth_study import phase_poly
from qiskit import QuantumCircuit
from math import comb
from experiments.qaoa_realistic import best as compile_best

CASES = [(10, 6, [5, 5]), (12, 8, [4, 4, 4])]
Dmax, Bmax = 3, 3


def dicke_block(size, l, s, co, params, pairs='path'):
    """Dicke(l) on the long qubits x Dicke(s) on the short qubits (2*size qubits, no registers) + objective phase + pair-swap mixers.
    Long/short exclusivity is NOT imposed by the circuit: it is invariant under the pair-swap mixer, so infeasible shots are discarded
    classically at the end (success rate = feasible / (C(m,l) C(m,s)))."""
    from negsearch.symlower import symmetric_block_prep
    qc = symmetric_block_prep(size, l, s)          # exact: support = F (exclusivity included), no registers, no post-selection
    p = len(params) // 2
    for k in range(p):
        phase_poly(qc, co, params[k]); swap_mixer_layer(qc, [q for j in range(size) for q in (2 * j, 2 * j + 1)], params[p + k], pairs)
    return qc


def feasible_count(B):
    from negsearch.large_case import enumerate_automaton
    return len(enumerate_automaton(B))


def per_period_options(sectors):
    combos = {}
    for size in set(sectors):
        combos[size] = []
        for l in range(2):
            for s in range(Bmax + 1):
                B = sector_block_automaton(size, l, s)
                if B.width[B.n] > 0:
                    c = feasible_count(B)
                    if c > 0:
                        combos[size].append((l, s, c))
    opts = []
    for choice in itertools.product(*[combos[sz] for sz in sectors]):
        L = sum(c[0] for c in choice); S = sum(c[1] for c in choice)
        if 0 <= L - S <= Dmax and L + S <= Bmax:
            w = 1
            for c in choice: w *= c[2]
            opts.append((tuple((c[0], c[1]) for c in choice), L + S, w))
    return opts


def sample_interface(opts, T, Q, rng):
    w = np.array([o[2] for o in opts], float); w /= w.sum()
    while True:
        idx = rng.choice(len(opts), size=T, p=w)
        if sum(opts[i][1] for i in idx) <= Q:
            return tuple(opts[i][0] for i in idx)


def quad_form(h, J, nv):
    M = np.diag(h).astype(float)
    for (i, j), v in J.items():
        M[i, j] += v / 2; M[j, i] += v / 2
    return M


def run_case(A, T, sectors, seed=5, n_iface=3, sweeps=3, shots=10, restarts=3):
    t0 = time.time()
    nv = 2 * A * T; Q = int(round(0.6 * T * Bmax))
    h, J = qubo_cost(A, T, seed)
    Jm = max(abs(v) for v in J.values())
    Jp = {k: v for k, v in J.items() if abs(v) >= 0.25 * Jm}
    opts = per_period_options(sectors)
    sec_off = np.cumsum([0] + list(sectors))
    rng = np.random.default_rng(1)
    res = dict(A=A, T=T, n=nv, sectors=sectors, J_terms=len(J), J_terms_pruned=len(Jp), interfaces=[])
    for it in range(n_iface):
        iface = sample_interface(opts, T, Q, rng)
        blocks = []
        for t in range(T):
            for si, size in enumerate(sectors):
                l, s = iface[t][si]
                strs = block_strings(sector_block_automaton(size, l, s))
                V = [x_index(A, t, sec_off[si] + j, d) for j in range(size) for d in (0, 1)]
                S = np.array([x for x, _ in strs], float)
                blocks.append(dict(t=t, si=si, size=size, ls=(l, s), V=V, strs=strs, S=S))
        out = dict(iface=[[list(c) for c in per] for per in iface], methods={})
        for label, (hh, JJ) in {'full': (h, J), 'pruned': (h, Jp)}.items():
            M = quad_form(hh, JJ, nv)
            Mtrue = quad_form(h, J, nv)

            def local(b, x, M=M):
                V = b['V']; R = [k for k in range(nv) if k not in set(V)]
                lin = 2 * M[np.ix_(V, R)] @ x[R]
                MVV = M[np.ix_(V, V)]
                return MVV, lin

            def cost_vec(b, x):
                MVV, lin = local(b, x)
                S = b['S']
                return np.einsum('ij,jk,ik->i', S, MVV, S) + S @ lin

            # --- compile one sweep
            comp = []
            x0 = np.zeros(nv)
            for b in blocks:
                # random feasible start for the fixed neighbours
                pass
            def rand_start():
                x = np.zeros(nv)
                for b in blocks:
                    j = rng.integers(len(b['strs'])); x[b['V']] = b['strs'][j][0]
                return x
            xr = rand_start()
            if label == 'pruned' or label == 'full':
                for b in blocks:
                    MVV, lin = local(b, xr); m2 = len(b['V'])
                    hl = np.diag(MVV) + lin
                    Jl = {(i, j): 2 * MVV[i, j] for i in range(m2) for j in range(i + 1, m2) if abs(MVV[i, j]) > 0}
                    co = ising_from_qubo(hl, Jl)
                    qc = dicke_block(b['size'], b['ls'][0], b['ls'][1], co, [0.3, 0.8])
                    r = compile_best(qc, seeds=3)
                    comp.append(dict(t=b['t'], si=b['si'], ls=list(b['ls']), terms=len(Jl), post_sel=len(b['strs']) / (comb(b['size'], b['ls'][0]) * comb(b['size'], b['ls'][1])), **r))
            out.setdefault('compile', {})[label] = comp

            # --- quality (exact simulation of each block p=1)
            def qaoa_block(b, x, noise_esp):
                cv = cost_vec(b, x); m = len(cv)
                strs = [(tuple(int(v) for v in row), 1.0 / len(b['strs'])) for row in b['S']]
                bl = [(b['t'], b['si'], strs)]; combos = np.arange(m)[:, None]
                def f(p):
                    pr = tensor_qaoa_swap(bl, combos, cv, list(p), pairs='path').ravel()
                    return float(pr @ cv)
                bestp = None
                for r in range(restarts):
                    p0 = rng.uniform(0, 1, 2)
                    o = minimize(f, p0, method='Nelder-Mead', options=dict(xatol=1e-3, fatol=1e-6, maxiter=80))
                    if bestp is None or o.fun < bestp.fun: bestp = o
                pr = tensor_qaoa_swap(bl, combos, cv, list(bestp.x), pairs='path').ravel()
                mu = np.ones(m) / m
                pr = noise_esp * pr + (1 - noise_esp) * mu        # noisy model: failed runs ~ feasible-uniform after post-selection
                pr = pr / pr.sum()
                return pr, cv

            def total(x):
                return float(x @ Mtrue @ x)

            def bcd(method, x, noise=None):
                x = x.copy()
                for sw in range(sweeps):
                    for bi, b in enumerate(blocks):
                        cv = cost_vec(b, x)
                        cur = int(np.argmin(np.abs(b['S'] - x[b['V']]).sum(1)))
                        if method == 'exact':
                            j = int(np.argmin(cv))
                        elif method == 'uniform':
                            smp = rng.choice(len(cv), size=shots); j = int(smp[np.argmin(cv[smp])])
                            if cv[j] > cv[cur]: j = cur
                        else:
                            esp = 1.0 if noise is None else noise[bi]
                            pr, _ = qaoa_block(b, x, esp)
                            smp = rng.choice(len(cv), size=shots, p=pr)
                            j = int(smp[np.argmin(cv[smp])])
                            if method == 'uniform':
                                smp = rng.choice(len(cv), size=shots); j = int(smp[np.argmin(cv[smp])])
                            if cv[j] > cv[cur]: j = cur
                        x[b['V']] = b['S'][j]
                return x
            esp_list = [c['esp'] for c in comp]
            rows = {}
            starts = [rand_start() for _ in range(2)]
            rnd = np.mean([total(rand_start()) for _ in range(300)])
            rows['random_feasible'] = rnd
            rows['exact_blocks'] = min(total(bcd('exact', s)) for s in [rand_start() for _ in range(30)])
            rows['uniform_same_shots'] = float(np.mean([total(bcd('uniform', s)) for s in starts]))
            rows['qaoa_ideal'] = float(np.mean([total(bcd('qaoa', s)) for s in starts]))
            rows['qaoa_noisy_model'] = float(np.mean([total(bcd('qaoa', s, esp_list)) for s in starts]))
            out['methods'][label] = rows
            print(f"A={A},T={T} iface#{it} [{label}] compiled sweep: CZ/block median {np.median([c['cz'] for c in comp]):.0f}, max {max(c['cz'] for c in comp)}, ESP median {np.median(esp_list):.2f} min {min(esp_list):.2f}; "
                  f"cost: random {rnd:.3f}, exact-blocks {rows['exact_blocks']:.3f}, uniform {rows['uniform_same_shots']:.3f}, QAOA ideal {rows['qaoa_ideal']:.3f}, QAOA noisy-model {rows['qaoa_noisy_model']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        res['interfaces'].append(out)
    return res


def main():
    allres = []
    for A, T, S in CASES:
        allres.append(run_case(A, T, S))
        json.dump(allres, open('results/block_decomposition_sym.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
