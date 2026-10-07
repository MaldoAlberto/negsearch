"""What can be pruned from our QAOA layer?  Cumulative stages, all compiled to the ibm_fez topology (nothing executed).
Exact stages (the state on the feasible set is unchanged up to a global phase; verified against a dense simulation for n<=16):
  S0  symmetric preparation + pair-exchange mixers (path) + full objective                 [baseline of qaoa_symlower]
  S1  + variables fixed by the interface (long variables of a block with l=0, short with s=0, ...) are eliminated from the objective:
        terms touching a variable known to be 0 vanish, terms with a variable known to be 1 become linear
  S2  + blocks with a single feasible string get no mixer; a block with l=0 (or s=0) uses a two-qubit swap rotation instead of the four-qubit one
  S3  + constraint gauge: on the feasible set the counts of every (block, long/short) group are constants, so a coupling J_ij can be replaced by
        J_ij - r_i - c_j (and J_ij - a_i - a_j inside a group) with an exact linear compensation; the exclusivity pairs are free (their product is 0)
Approximate stages (the objective changes; quality measured on the exact feasible set of the interface tuple, p=1 angles optimised):
  S4/S5/S6  + drop couplings below 10 / 25 / 50 % of the largest remaining coefficient.
Writes results/prune_circuit.json."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)
import json, math, warnings; warnings.filterwarnings('ignore')
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.circuit.library import RXXGate, RYYGate, RZZGate
from qiskit.quantum_info import Statevector
from negsearch.hw_large import interface_tuples, qubo_cost, ising_from_qubo, x_index, pair_swap_mixer, swap_pairs, block_strings
from negsearch.depth_study import phase_poly
from negsearch.symlower import symmetric_block_prep
from negsearch.large_case import sector_block_automaton
from experiments.qaoa_realistic import best as compile_best

CASES = [(4, 1, [2, 2]), (6, 1, [3, 3]), (4, 2, [2, 2]), (6, 2, [3, 3])]


def blocks_of(A, T, S, iface):
    off = np.cumsum([0] + list(S)); bl = []
    for t in range(T):
        for si, size in enumerate(S):
            l, s = iface[t][si]; xq = [x_index(A, t, off[si] + j, d) for j in range(size) for d in (0, 1)]
            strs = np.array([x for x, _ in block_strings(sector_block_automaton(size, l, s))])
            bl.append(dict(xq=xq, size=size, l=l, s=s, F=strs))
    return bl


def forced_values(bl):
    f = {}
    for b in bl:
        for p, v in enumerate(b['xq']):
            col = b['F'][:, p]
            if col.min() == col.max(): f[v] = int(col[0])
    return f


def eliminate(h, J, forced):
    h = h.copy(); J2 = {}
    for (i, j), v in J.items():
        fi, fj = forced.get(i), forced.get(j)
        if fi is not None and fj is not None: continue
        if fi is not None:
            if fi: h[j] += v
        elif fj is not None:
            if fj: h[i] += v
        else: J2[(i, j)] = J2.get((i, j), 0) + v
    for i in forced: h[i] = 0.0
    return h, J2


def groups_of(bl, forced):
    g = []
    for b in bl:
        for d, cnt in ((0, b['l']), (1, b['s'])):
            vs = [b['xq'][2 * j + d] for j in range(b['size'])]
            n1 = sum(1 for v in vs if forced.get(v) == 1)
            free = [v for v in vs if v not in forced]
            if free: g.append((free, cnt - n1))
    return g


def gauge(h, J, groups, partner):
    """exact reduction on the feasible set; partner[v] = the exclusive variable of v (or None)"""
    h = h.copy(); J = dict(J)
    def get(i, j): return J.get((min(i, j), max(i, j)), 0.0)
    def put(i, j, v):
        k = (min(i, j), max(i, j))
        if abs(v) < 1e-14: J.pop(k, None)
        else: J[k] = v
    for gi, (vs, nk) in enumerate(groups):
        if len(vs) < 3: continue
        pairs = [(a, b) for a in range(len(vs)) for b in range(a + 1, len(vs))]
        M = np.zeros((len(pairs), len(vs) + 1)); y = np.zeros(len(pairs))
        for r, (a, b) in enumerate(pairs): M[r, a] = 1; M[r, b] = 1; M[r, -1] = 1; y[r] = get(vs[a], vs[b])
        sol = np.linalg.lstsq(M, y, rcond=None)[0]; a_ = sol[:-1]
        for (a, b) in pairs: put(vs[a], vs[b], get(vs[a], vs[b]) - a_[a] - a_[b] - sol[-1])
        for a, v in enumerate(vs): h[v] += (nk - 1) * a_[a]
    for gi in range(len(groups)):
        for gj in range(gi + 1, len(groups)):
            (vi, ni), (vj, nj) = groups[gi], groups[gj]
            if len(vi) + len(vj) < 3: continue
            rows = [(a, b) for a in range(len(vi)) for b in range(len(vj)) if partner.get(vi[a]) != vj[b]]
            if not rows: continue
            M = np.zeros((len(rows), len(vi) + len(vj) + 1)); y = np.zeros(len(rows))
            for r, (a, b) in enumerate(rows): M[r, a] = 1; M[r, len(vi) + b] = 1; M[r, -1] = 1; y[r] = get(vi[a], vj[b])
            sol = np.linalg.lstsq(M, y, rcond=None)[0]; r_ = sol[:len(vi)]; c_ = sol[len(vi):-1]
            for (a, b) in [(a, b) for a in range(len(vi)) for b in range(len(vj))]:
                if partner.get(vi[a]) == vj[b]: put(vi[a], vj[b], 0.0)            # exclusive pair: product is 0 on F, any value is exact
                else: put(vi[a], vj[b], get(vi[a], vj[b]) - r_[a] - c_[b] - sol[-1])
            for a, v in enumerate(vi): h[v] += nj * r_[a]
            for b, v in enumerate(vj): h[v] += ni * c_[b]
    return h, J


def prune_J(J, h, frac):
    if not J or not frac: return J
    m = max(abs(v) for v in J.values())
    return {k: v for k, v in J.items() if abs(v) >= frac * m}


def swap_rot(qc, a, b, beta):
    """exp(-i beta SWAP(a,b)) up to a global phase"""
    qc.append(RXXGate(beta), [a, b]); qc.append(RYYGate(beta), [a, b]); qc.append(RZZGate(beta), [a, b])


def circuit(A, T, S, bl, h, J, gam, bet, smart_mixers):
    nx = 2 * A * T; qc = QuantumCircuit(nx)
    for b in bl: qc.compose(symmetric_block_prep(b['size'], b['l'], b['s']), qubits=b['xq'], inplace=True)
    co = ising_from_qubo(h, J)
    for g, be in zip(gam, bet):
        phase_poly(qc, co, g)
        for b in bl:
            m = b['size']; xq = b['xq']
            if smart_mixers and len(b['F']) == 1: continue
            for j, k in swap_pairs(m, 'path'):
                if smart_mixers and b['l'] == 0: swap_rot(qc, xq[2 * j + 1], xq[2 * k + 1], be)
                elif smart_mixers and b['s'] == 0: swap_rot(qc, xq[2 * j], xq[2 * k], be)
                else: pair_swap_mixer(qc, xq[2 * j], xq[2 * j + 1], xq[2 * k], xq[2 * k + 1], be)
    return qc


def dense_state(A, T, S, bl, h, J, gam, bet):
    """independent dense simulation on the full space (n<=16): uniform on F, e^{-i g c}, pair-exchange rotations"""
    nx = 2 * A * T; N = 2 ** nx; idx = np.arange(N); bits = ((idx[:, None] >> np.arange(nx)) & 1)
    feas = np.ones(N, bool)
    for b in bl:
        xq = b['xq']; L = bits[:, xq[0::2]].sum(1); Sh = bits[:, xq[1::2]].sum(1); ex = (bits[:, xq[0::2]] & bits[:, xq[1::2]]).sum(1)
        feas &= (L == b['l']) & (Sh == b['s']) & (ex == 0)
    return bits, feas


def run_dense(A, T, S, bl, bits, feas, phase, gam, bet):
    N = len(feas); idx = np.arange(N); psi = (feas / math.sqrt(feas.sum())).astype(complex)
    perms = []
    for b in bl:
        for j, k in swap_pairs(b['size'], 'path'):
            perm = idx.copy(); xq = b['xq']
            for a_, b_ in ((xq[2 * j], xq[2 * k]), (xq[2 * j + 1], xq[2 * k + 1])):
                ba = (perm >> a_) & 1; bb = (perm >> b_) & 1; perm = np.where(ba == bb, perm, perm ^ ((1 << a_) | (1 << b_)))
            perms.append(perm)
    for g, be in zip(gam, bet):
        psi = psi * np.exp(-1j * g * phase)
        for perm in perms: psi = math.cos(be) * psi - 1j * math.sin(be) * psi[perm]
    return psi


def cost_vec(bits, h, J):
    c = bits @ h
    for (i, j), v in J.items(): c = c + v * bits[:, i] * bits[:, j]
    return c


def main():
    out = []; rng = np.random.default_rng(3)
    print(f"{'n':>3s} {'stage':34s} {'ZZ':>4s} {'CZ':>5s} {'depth':>6s} {'ESP':>6s}  {'exact?':>7s}  {'quality':>7s}")
    for A, T, S in CASES:
        n = 2 * A * T; h0, J0 = qubo_cost(A, T, 5)
        tup, _ = interface_tuples(A, T, S); iface = max(tup, key=lambda x: x[1])[0]
        bl = blocks_of(A, T, S, iface); forced = forced_values(bl)
        partner = {}
        for b in bl:
            for j in range(b['size']): partner[b['xq'][2 * j]] = b['xq'][2 * j + 1]; partner[b['xq'][2 * j + 1]] = b['xq'][2 * j]
        h1, J1 = eliminate(h0, J0, forced)
        h3, J3 = gauge(h1, J1, groups_of(bl, forced), partner)
        stages = [('S0 baseline (full objective)', h0, J0, False, True), ('S1 + eliminate forced variables', h1, J1, False, True),
                  ('S2 + trivial blocks, 2-qubit mixers', h1, J1, True, True), ('S3 + constraint gauge', h3, J3, True, True)]
        for fr in (0.10, 0.25, 0.50):
            stages.append((f'S{4 if fr == 0.10 else 5 if fr == 0.25 else 6} + prune {int(fr*100)}% of the largest', h3, prune_J(J3, h3, fr), True, False))
        # exact quality machinery (n<=16)
        dense = None; dense_cache = {}
        if n <= 16:
            bits, feas = dense_state(A, T, S, bl, h0, J0, None, None); ctrue = cost_vec(bits.astype(float), h0, J0)
            cmin, cmax = ctrue[feas].min(), ctrue[feas].max(); dense = (bits.astype(float), feas, ctrue, cmin, cmax)
        for name, hh, JJ, smart, exact in stages:
            qc = circuit(A, T, S, bl, hh, JJ, [0.3], [0.8], smart); r = compile_best(qc, seeds=6)
            row = dict(n=n, stage=name, zz=len(JJ), cz=r['cz'], depth=r['depth'], esp=r['esp'], qubits=r['qubits'])
            if dense is not None:
                bits, feas, ctrue, cmin, cmax = dense
                # exactness check: circuit (Qiskit statevector) against dense simulation with the ORIGINAL objective, random angles, up to a global phase
                if exact:
                    gam = rng.uniform(-2, 2, 2); bet = rng.uniform(0, 3, 2)
                    psi = Statevector(circuit(A, T, S, bl, hh, JJ, gam, bet, smart)).data
                    ref = run_dense(A, T, S, bl, bits, feas, cost_vec(bits, h0, J0), gam, bet)
                    row['exact_overlap_defect'] = float(1 - abs(np.vdot(ref, psi)))
                # quality with the phase that the circuit uses, true cost for the score (p=1).  Exact stages have the same state up to a global phase,
                # hence the same quality as S0 (re-optimising would only add optimiser noise); approximate stages start from the S0 angles + restarts.
                ph = cost_vec(bits, hh, JJ)
                def f(x):
                    P = np.abs(run_dense(A, T, S, bl, bits, feas, ph, [x[0]], [x[1]])) ** 2
                    return -float((P * feas * (1 - (ctrue - cmin) / (cmax - cmin))).sum())
                if exact and 'q0' in dense_cache:
                    row['quality'] = dense_cache['q0']
                else:
                    starts = ([dense_cache['x0']] if 'x0' in dense_cache else []) + [rng.uniform(0, 1.5, 2) for _ in range(6)]
                    bestq = min((minimize(f, x0, method='Nelder-Mead', options=dict(maxiter=120, xatol=1e-3, fatol=1e-6)) for x0 in starts), key=lambda o: o.fun)
                    row['quality'] = -float(bestq.fun)
                    if name.startswith('S0'): dense_cache['q0'] = row['quality']; dense_cache['x0'] = bestq.x
            out.append(row)
            ex = f"{row['exact_overlap_defect']:.0e}" if 'exact_overlap_defect' in row else ('exact' if exact else 'approx')
            print(f"{n:3d} {name:34s} {row['zz']:4d} {row['cz']:5d} {row['depth']:6d} {row['esp']:6.3f}  {ex:>7s}  {row.get('quality', float('nan')):7.3f}", flush=True)
    json.dump(out, open('results/prune_circuit.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
