"""Fusing the objective into the swap mixers.  (a) brick-wall order of the pair exchanges (odd pairs of the path, then even pairs) instead of the sequential path: depth m-1 -> 2 per block;
(b) for blocks whose mixer is a two-qubit swap rotation (no long or no short position) the ZZ phase between the swapped variables commutes with that swap, so with the odd pairs
(which act first and are disjoint) the phase is absorbed:  exp(-i beta SWAP) exp(-i g c ZZ) = RXX(b) RYY(b) RZZ(b + 2 g c)  -> 3 CZ instead of 3 + 2.  The state is identical to (a) without fusion.
Compared on the n=64, n=100 and n=192 cores (compiled to ibm_fez), plus ideal quality of the sequential vs brick-wall mixer inside F.  Writes results/fused_mixer_<tag>.json."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)
import json, math, os, warnings; warnings.filterwarnings('ignore')
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import experiments.case_four as C
import experiments.prune_circuit as P
from negsearch.hw_large import ising_from_qubo, swap_pairs, pair_swap_mixer
from negsearch.depth_study import phase_poly
from negsearch.symlower import symmetric_block_prep
from experiments.qaoa_realistic import best as compile_best

cp = C.core_problem(); bl, h1, J1, col, Fm = cp['bl'], cp['h1'], cp['J1'], cp['col'], cp['F']; n = C.n; f = len(cp['free']); nF = len(Fm)


def order_pairs(m, brick):
    pr = swap_pairs(m, 'path')
    return [p for i, p in enumerate(pr) if i % 2 == 0] + [p for i, p in enumerate(pr) if i % 2 == 1] if brick else pr


def build(g, b, brick, fuse):
    qc = QuantumCircuit(n)
    for blk in bl: qc.compose(symmetric_block_prep(blk['size'], blk['l'], blk['s']), qubits=blk['xq'], inplace=True)
    co = ising_from_qubo(h1, J1); fused = {}
    plan = []
    for blk in bl:
        m = blk['size']; xq = blk['xq']
        if len(blk['F']) == 1: continue
        two = blk['l'] == 0 or blk['s'] == 0; d = 1 if blk['l'] == 0 else 0
        for idx, (j, k) in enumerate(order_pairs(m, brick)):
            first = (not brick) or idx < len(swap_pairs(m, 'path')[::2])           # odd-position pairs act first in brick order (disjoint)
            if two:
                a, c = xq[2 * j + d], xq[2 * k + d]
                if fuse and brick and first and (min(a, c), max(a, c)) in co: fused[(a, c)] = co.pop((min(a, c), max(a, c)))
                plan.append(('swap', a, c))
            else: plan.append(('pair', xq[2 * j], xq[2 * j + 1], xq[2 * k], xq[2 * k + 1]))
    phase_poly(qc, co, g)
    for it in plan:
        if it[0] == 'swap':
            a, c = it[1], it[2]; extra = 2 * g * fused.get((min(a, c), max(a, c)), 0.0)
            qc.rxx(b, a, c); qc.ryy(b, a, c); qc.rzz(b + extra, a, c)
        else: pair_swap_mixer(qc, it[1], it[2], it[3], it[4], b)
    return qc


out = {'n': n, 'f': f, 'nF': nF}
qs = {k: build(0.37, 0.61, *v) for k, v in {'sequential': (False, False), 'brick': (True, False), 'brick+fused': (True, True)}.items()}
if f <= 20:
    gc = {k: C.compact(q) for k, q in qs.items()}
    sv = {k: Statevector(v[0]).data for k, v in gc.items()}
    # qubit orders may differ between compactions only if the used sets differ; compare brick vs brick+fused (same qubit set)
    if gc['brick'][1] == gc['brick+fused'][1]: out['fused_equals_brick_overlap_defect'] = float(1 - abs(np.vdot(sv['brick'], sv['brick+fused'])))
    print('brick vs brick+fused defect', out.get('fused_equals_brick_overlap_defect'), flush=True)
for k, q in qs.items():
    r = compile_best(C.compact(q)[0], seeds=3); out[k] = r
    print(f"{k:12s} CZ {r['cz']:5d} depth {r['depth']:5d} ESP {r['esp']:.3g}", flush=True)
# quality of sequential vs brick-wall order, inside F (ideal)
cF = C.cost_of(Fm, h1, J1, col); cmin, cmax = cF.min(), cF.max(); cn = (cF - cmin) / (cmax - cmin); opt = np.abs(cF - cmin) < 1e-9
if nF <= 400000:
    idx_of = {tuple(r): i for i, r in enumerate(Fm)}

    def perms(brick):
        L = []
        for blk in bl:
            if len(blk['F']) == 1: continue
            for (j, k) in order_pairs(blk['size'], brick):
                cols = [(col[blk['xq'][2 * j + d]], col[blk['xq'][2 * k + d]]) for d in (0, 1) if blk['xq'][2 * j + d] in col]; pm = np.empty(nF, int)
                for i, r in enumerate(Fm):
                    r2 = r.copy()
                    for a_, c_ in cols: r2[a_], r2[c_] = r[c_], r[a_]
                    pm[i] = idx_of[tuple(r2)]
                L.append(pm)
        return L
    for brick in (False, True):
        PL = perms(brick)
        def st(gm, bt):
            psi = np.ones(nF, complex) / math.sqrt(nF) * np.exp(-1j * gm * cF)
            for pm in PL: psi = math.cos(bt) * psi - 1j * math.sin(bt) * psi[pm]
            return psi
        rng = np.random.default_rng(0); bo = None
        for _ in range(20):
            o = minimize(lambda x: float((np.abs(st(*x)) ** 2 * cF).sum()), [rng.uniform(-2, 2), rng.uniform(0, 1.6)], method='Nelder-Mead')
            if bo is None or o.fun < bo.fun: bo = o
        Pf = np.abs(st(*bo.x)) ** 2
        out['quality_' + ('brick' if brick else 'sequential')] = dict(quality=float((Pf * (1 - cn)).sum()), popt=float(Pf[opt].sum()))
        print('quality', 'brick' if brick else 'sequential', out['quality_' + ('brick' if brick else 'sequential')], flush=True)
json.dump(out, open(f"results/fused_mixer_{os.environ.get('CF_TAG', 'n64')}.json", 'w'), indent=1, default=float)
