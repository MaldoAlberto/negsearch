"""~100-qubit hardware experiment: many independent feasible-by-construction blocks run in parallel on one chip.

Setting (same family as the paper's residual-core study).  A portfolio with A=10 assets, T=6 periods, 2 sectors of 5 assets, long/short
variables -> n = 2*A*T = 120 binary variables.  The interface (per period and sector counts (l, s) of longs and shorts) is drawn classically;
given it, the problem splits into 12 blocks of 10 qubits (period t, sector).  Each block is a small QUBO (the neighbours enter as linear fields).
All 12 blocks run in ONE circuit of 120 qubits on disjoint connected regions of the chip.

Methods (every one acts on the same block QUBO, same shots, same regions):
  ours       p=1 : exact support-F preparation (Dicke on longs x Dicke on shorts, controlled-swap decompression) + objective phase + pair-exchange mixers
  ours_p0    p=0 : preparation only (control: what the hardware does to a feasible state)
  uniform        : Hadamards (control: feasible fraction |F|/2^10 of an unconstrained sample)
  lcu            : single-basis Fourier-LCU variant: warm start RY(2 asin sqrt(k/m)) per qubit (the start of the Fourier-LCU paper), objective + one Rz per count constraint
                   + exclusivity phase, X mixer, angles trained
  unbalanced     : same warm start, objective + (-l1 g + l2 g^2) for each count constraint + exclusivity penalty, X mixer, angles trained
  xy             : same warm start, objective + exclusivity phase, XY mixer inside every count group (the coherent circuit the LCU approximates), angles trained
The baselines are given the SAME block decomposition and the SAME count constraints (the favourable setting for them).
Nothing here needs the IBM key except `connect()`; the key is read from a file that is never part of the repository."""
from __future__ import annotations
import os as _os0
_os0.environ.setdefault('NS_OPT_PREP', '1')      # optimised Dicke + pruned decompression (same state); set to 0 to reproduce v21

import itertools
import json
import math
import os
import sys
from collections import defaultdict
from math import comb

import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit.transpiler import CouplingMap

from .hw_large import qubo_cost, ising_from_qubo, x_index, pair_swap_mixer
from .depth_study import phase_poly
from .symlower import symmetric_block_prep
from . import tilt as TILT

A, T, M, NB_Q = 10, 6, 5, 10          # assets, periods, assets per sector, qubits per block
SECT = [5, 5]
METHODS = ['ours', 'ours_full', 'ours_p0', 'uniform', 'lcu', 'unbalanced', 'xy', 'ours_tilt_p0', 'ours_tilt', 'lcu_b', 'unbalanced_b', 'xy_b']
BASE_METHODS = ['ours', 'ours_full', 'ours_p0', 'uniform', 'lcu', 'unbalanced', 'xy']      # v21-v23 set; *_tilt and *_b use the classical tilt (see negsearch.tilt)
TILT_TAU = 4.0
BASIS = ['cz', 'rz', 'sx', 'x']
# Variants of our p=1 layer.  'pairs' = asset pairs exchanged by the mixer (every pair keeps feasibility; fewer pairs = fewer gates),
# 'keep' = fraction of the largest objective couplings kept in OUR objective layer (the support, hence feasibility, is unaffected).
VARIANTS = {'ours': dict(pairs=[(0, 1), (2, 3), (1, 2)], keep=0.5),            # light: what the noise allows
            'ours_full': dict(pairs=[(0, 1), (1, 2), (2, 3), (3, 4)], keep=1.0)}  # full objective, full path of exchanges


# ----------------------------------------------------------------------------------------------------------------- instance
def strings_all():
    return np.array(list(itertools.product([0, 1], repeat=NB_Q)), dtype=np.int8)   # row i <-> (q0..q9), axis 0 = q0


S_ALL = strings_all()


def feasible_mask(l, s):
    L = S_ALL[:, 0::2]; Sh = S_ALL[:, 1::2]
    return (L.sum(1) == l) & (Sh.sum(1) == s) & ((L & Sh).sum(1) == 0)


def quad_form(h, J):
    n = len(h); Mx = np.diag(h).astype(float)
    for (i, j), v in J.items():
        Mx[i, j] += v / 2; Mx[j, i] += v / 2
    return Mx


def build_instance(seed_obj=5, seed_iface=11, family='medium', min_F=10, qfrac=0.75):
    """classical part: objective, interface tuple (all 12 blocks non-trivial), per-block local QUBO with a fixed random feasible neighbourhood"""
    from experiments.core_map import period_options, FAMILIES, nF
    nv = 2 * A * T
    h, J = qubo_cost(A, T, seed_obj)
    Mx = quad_form(h, J)
    p = FAMILIES[family]; opts = period_options(M, 2, p); Q = int(round(qfrac * T * p['Bmax']))
    rng = np.random.default_rng(seed_iface)
    w = np.array([o[2] for o in opts], float); w /= w.sum()
    iface = None
    for _ in range(200000):
        pick = [opts[i] for i in rng.choice(len(opts), T, p=w)]
        if sum(o[1] for o in pick) > Q: continue
        if all(nF(M, l, s) >= min_F for o in pick for (l, s) in o[0]):
            iface = [list(map(tuple, o[0])) for o in pick]; break
    assert iface is not None, 'no interface with 12 blocks of at least min_F feasible strings'
    sec_off = np.cumsum([0] + SECT)
    blocks = []
    for t in range(T):
        for si, size in enumerate(SECT):
            l, s = iface[t][si]
            V = [x_index(A, t, int(sec_off[si]) + j, d) for j in range(size) for d in (0, 1)]
            blocks.append(dict(id=len(blocks), t=t, si=si, ls=(l, s), V=V, mask=feasible_mask(l, s)))
    # fixed feasible neighbourhood (random feasible string per block): linear fields for every block
    x_fix = np.zeros(nv)
    for b in blocks:
        idx = rng.choice(np.flatnonzero(b['mask'])); x_fix[b['V']] = S_ALL[idx]
    for b in blocks:
        V = b['V']; R = [k for k in range(nv) if k not in set(V)]
        MVV = Mx[np.ix_(V, V)]; lin = 2 * Mx[np.ix_(V, R)] @ x_fix[R]
        hl = np.diag(MVV) + lin
        Jl = {(i, j): 2 * MVV[i, j] for i in range(NB_Q) for j in range(i + 1, NB_Q) if abs(MVV[i, j]) > 0}
        b['hl'] = hl; b['Jl'] = Jl
        b['cv'] = qubo_energy(S_ALL, hl, Jl)               # local cost of every string (neighbours fixed)
        cf = b['cv'][b['mask']]
        b['cmin'], b['cmax'], b['cmean'] = float(cf.min()), float(cf.max()), float(cf.mean())
    return dict(A=A, T=T, nv=nv, M=Mx, h=h, J=J, iface=iface, blocks=blocks, x_fix=x_fix, seed_obj=seed_obj, seed_iface=seed_iface, family=family)


def qubo_energy(S, h, J):
    e = S @ np.asarray(h, float)
    for (i, j), v in J.items():
        e = e + v * S[:, i] * S[:, j]
    return e


# ----------------------------------------------------------------------------------------------------------------- baselines (qubo form)
def penalty_qubo(l, s, method, par):
    """(h, J) of the constraint terms of the baselines on the 10 block qubits.  par = parameters of the method (after gamma, beta)"""
    method = method[:-2] if method.endswith('_b') else method
    h = np.zeros(NB_Q); J = {}
    groups = [(list(range(0, NB_Q, 2)), l), (list(range(1, NB_Q, 2)), s)]
    if method == 'xy':                        # exclusivity only; the counts are kept by the XY mixer
        lam_ex = par[0]
    elif method == 'lcu':                       # single-basis: one Rz angle per count constraint (linear), coherent exclusivity pair phase
        th = par[:2]; lam_ex = par[2]
        for (G, c), t in zip(groups, th):
            for i in G: h[i] += t
    else:                                     # unbalanced: -l1 g + l2 g^2
        l1, l2, lam_ex = par[:3]
        for G, c in groups:
            for i in G: h[i] += -l1 + l2 * (1 - 2 * c)
            for a, b in itertools.combinations(G, 2): J[(a, b)] = J.get((a, b), 0) + 2 * l2
    for j in range(NB_Q // 2): J[(2 * j, 2 * j + 1)] = J.get((2 * j, 2 * j + 1), 0) + lam_ex
    return h, J


def baseline_qubo(b, method, par):
    gam = par[0]; ph, pJ = penalty_qubo(*b['ls'], method, par[2:])
    h = gam * b['hl'] + ph
    J = {k: gam * v for k, v in b['Jl'].items()}
    for k, v in pJ.items(): J[k] = J.get(k, 0) + v
    return h, J


def rx_all(psi, beta):
    c, s = math.cos(beta), -1j * math.sin(beta)
    for ax in range(NB_Q):
        a0 = np.take(psi, 0, axis=ax); a1 = np.take(psi, 1, axis=ax)
        psi = np.stack([c * a0 + s * a1, s * a0 + c * a1], axis=ax)
    return psi


XY_PAIRS = [(a, a + 2) for a in range(0, NB_Q - 2, 2)] + [(a, a + 2) for a in range(1, NB_Q - 2, 2)]      # paths inside the long group and inside the short group


def xy_all(psi, beta):
    """exp(-i beta (XX+YY)/2) on every pair of XY_PAIRS, sequentially (axis k = qubit k)"""
    t = psi.copy(); c, s = math.cos(beta), -1j * math.sin(beta)
    for (a, b) in XY_PAIRS:
        i01 = [slice(None)] * NB_Q; i10 = [slice(None)] * NB_Q
        i01[a], i01[b] = 0, 1; i10[a], i10[b] = 1, 0
        i01, i10 = tuple(i01), tuple(i10)
        x01, x10 = t[i01].copy(), t[i10].copy()
        t[i01] = c * x01 + s * x10; t[i10] = c * x10 + s * x01
    return t


def warm_p(b, biased=False):
    if biased: return list(b['tilt']['marg'])
    l, s = b['ls']; return [l / M if k % 2 == 0 else s / M for k in range(NB_Q)]


def warm_state(b, biased=False):
    vecs = [np.array([math.sqrt(1 - p), math.sqrt(p)], complex) for p in warm_p(b, biased)]
    psi = vecs[0]
    for v in vecs[1:]: psi = np.multiply.outer(psi, v)
    return psi


def baseline_probs(b, method, par):
    h, J = baseline_qubo(b, method, par)
    E = qubo_energy(S_ALL, h, J).reshape((2,) * NB_Q)
    biased = method.endswith('_b'); base = method[:-2] if biased else method
    psi = warm_state(b, biased) * np.exp(-1j * E)
    psi = xy_all(psi, par[1]) if base == 'xy' else rx_all(psi, par[1])
    return (np.abs(psi) ** 2).reshape(-1)


def pruned_Jl(b, keep):
    Jl = b['Jl']
    if keep >= 1 or not Jl: return Jl
    mags = sorted((abs(v) for v in Jl.values()), reverse=True); thr = mags[max(0, int(math.ceil(keep * len(mags))) - 1)]
    return {k: v for k, v in Jl.items() if abs(v) >= thr}


def ours_probs(b, par, variant='ours'):
    """exact p=1 output of the 'ours' circuit: support-F state, phase of (pruned) objective, pair-exchange mixers on the listed asset pairs"""
    pairs = VARIANTS[variant]['pairs']; keep = VARIANTS[variant]['keep']
    idx = np.flatnonzero(b['mask']); ix = {tuple(int(v) for v in S_ALL[i]): k for k, i in enumerate(idx)}
    cph = qubo_energy(S_ALL[idx], b['hl'], pruned_Jl(b, keep))
    psi = np.full(len(idx), 1 / math.sqrt(len(idx)), dtype=complex)
    psi = psi * np.exp(-1j * par[0] * cph)
    for (j, k) in pairs:
        perm = []
        for i in idx:
            s_ = list(S_ALL[i]); s_[2 * j:2 * j + 2], s_[2 * k:2 * k + 2] = S_ALL[i][2 * k:2 * k + 2], S_ALL[i][2 * j:2 * j + 2]
            perm.append(ix[tuple(int(v) for v in s_)])
        inv = np.empty(len(perm), int); inv[np.array(perm)] = np.arange(len(perm))
        psi = math.cos(par[1]) * psi - 1j * math.sin(par[1]) * psi[inv]
    out = np.zeros(2 ** NB_Q); out[idx] = np.abs(psi) ** 2
    return out


def metrics(b, P):
    m = b['mask']; feas = float(P[m].sum())
    cf = b['cv'][m]; pf = P[m] / max(feas, 1e-300)
    mean = float(pf @ cf); span = b['cmax'] - b['cmin']
    opt_idx = np.flatnonzero(m)[np.argmin(cf)]
    return dict(feas=feas, quality=1 - (mean - b['cmin']) / span, p_opt=float(P[opt_idx]),
                uniform_quality=1 - (b['cmean'] - b['cmin']) / span, uniform_feas=float(m.mean()))


# ----------------------------------------------------------------------------------------------------------------- training
def train_ours(b, variant='ours', restarts=8, seed=0):
    rng = np.random.default_rng(seed); idx = np.flatnonzero(b['mask']); cv = b['cv'][idx]
    f = lambda p: float(ours_probs(b, p, variant)[idx] @ cv)
    best = None
    for r in range(restarts):
        o = minimize(f, rng.uniform(0, 1, 2), method='Nelder-Mead', options=dict(xatol=1e-3, fatol=1e-7, maxiter=120))
        if best is None or o.fun < best.fun: best = o
    return [float(v) for v in best.x]


def train_baseline(b, method, restarts=10, seed=0):
    rng = np.random.default_rng(seed)
    npar = {'lcu': 5, 'unbalanced': 5, 'xy': 3}[method[:-2] if method.endswith('_b') else method]      # gamma, beta, (th_L, th_S, lam_ex) | (l1, l2, lam_ex) | (lam_ex)

    def score(p):
        P = baseline_probs(b, method, p); mt = metrics(b, P)
        return -(mt['feas'] * mt['quality'])           # feasible mass times quality among feasible samples
    best = None
    for r in range(restarts):
        x0 = rng.uniform(-1.5, 1.5, npar)
        o = minimize(score, x0, method='Nelder-Mead', options=dict(xatol=1e-3, fatol=1e-8, maxiter=400))
        if best is None or o.fun < best.fun: best = o
    return [float(v) for v in best.x]


def add_tilt(b, tau=TILT_TAU):
    """classical information of the block: mean-field tilt of the feasible set, marginals, and the fitted Dicke angles (negsearch.tilt)"""
    idx = np.flatnonzero(b['mask']); X = S_ALL[idx].astype(float)
    g = TILT.mean_field_g(b['hl'], pruned_Jl(b, 1.0), X); w = TILT.tilt_weights(X, g, tau)
    pidx = (S_ALL[idx].astype(int) * (1 << np.arange(NB_Q))).sum(1)
    ang, cost = TILT.fit_angles(M, *b['ls'], pidx, w)
    b['tilt'] = dict(tau=tau, marg=np.clip(w @ X, 1e-3, 1 - 1e-3).tolist(), angles=[float(a) for a in ang], fit_cost=cost, target=w.tolist())


def train_tilt(b, par, restarts=5, seed=0):
    """gamma, beta of the p=1 layer on the tilted preparation, trained on the actual circuit state"""
    rng = np.random.default_rng(seed); m = b['mask']
    def f(p):
        pr = probs_from_circuit(block_circuit(b, 'ours_tilt', dict(par, ours_tilt=list(p)))); return float(pr[m] @ b['cv'][m] / pr[m].sum())
    best = None
    for r in range(restarts):
        o = minimize(f, rng.uniform(0, 1, 2), method='Nelder-Mead', options=dict(xatol=1e-3, fatol=1e-7, maxiter=100))
        if best is None or o.fun < best.fun: best = o
    return [float(v) for v in best.x]


def train_all(inst, seed=0, log=print):
    out = []
    for b in inst['blocks']:
        r = dict(ours=train_ours(b, 'ours', seed=seed), ours_full=train_ours(b, 'ours_full', seed=seed), lcu=train_baseline(b, 'lcu', seed=seed), unbalanced=train_baseline(b, 'unbalanced', seed=seed), xy=train_baseline(b, 'xy', seed=seed))
        if 'tilt' not in b: add_tilt(b)
        r['ours_tilt'] = train_tilt(b, r)
        for m_ in ('lcu_b', 'unbalanced_b', 'xy_b'): r[m_] = train_baseline(b, m_, seed=seed)
        out.append(r)
        log(f"block {b['id']:2d} (t={b['t']}, sector {b['si']}, l,s={b['ls']}, |F|={int(b['mask'].sum()):3d}) trained")
    return out


def ideal_probs(inst, params):
    """ideal output distribution over the 1024 strings of every block, for every method"""
    out = []
    for b, par in zip(inst['blocks'], params):
        d = {}
        idx = np.flatnonzero(b['mask']); u = np.zeros(2 ** NB_Q); u[idx] = 1.0 / len(idx)
        d['ours_p0'] = u
        d['ours'] = ours_probs(b, par['ours'], 'ours'); d['ours_full'] = ours_probs(b, par['ours_full'], 'ours_full')
        d['uniform'] = np.full(2 ** NB_Q, 2.0 ** -NB_Q)
        d['lcu'] = baseline_probs(b, 'lcu', par['lcu'])
        d['unbalanced'] = baseline_probs(b, 'unbalanced', par['unbalanced']); d['xy'] = baseline_probs(b, 'xy', par['xy'])
        if 'ours_tilt' in par:
            d['ours_tilt_p0'] = probs_from_circuit(block_circuit(b, 'ours_tilt_p0', par)); d['ours_tilt'] = probs_from_circuit(block_circuit(b, 'ours_tilt', par))
            for m_ in ('lcu_b', 'unbalanced_b', 'xy_b'): d[m_] = baseline_probs(b, m_, par[m_])
        out.append(d)
    return out


# ----------------------------------------------------------------------------------------------------------------- circuits (one block)
def block_circuit(b, method, par):
    """virtual 10-qubit circuit of one block; qubit k = variable b['V'][k]; measures qubit k into clbit k"""
    l, s = b['ls']
    qc = QuantumCircuit(NB_Q, NB_Q)
    if method in ('ours', 'ours_full', 'ours_p0'):
        qc.compose(symmetric_block_prep(M, l, s), inplace=True)
        if method != 'ours_p0':
            v = VARIANTS[method]
            co = ising_from_qubo(b['hl'], pruned_Jl(b, v['keep']))
            phase_poly(qc, co, par[method][0])
            for (j, k) in v['pairs']:
                pair_swap_mixer(qc, 2 * j, 2 * j + 1, 2 * k, 2 * k + 1, par[method][1])
    elif method in ('ours_tilt', 'ours_tilt_p0'):
        qc.compose(TILT.prep_with(M, l, s, b['tilt']['angles'])[0], inplace=True)
        if method == 'ours_tilt':
            v = VARIANTS['ours']
            co = ising_from_qubo(b['hl'], pruned_Jl(b, v['keep']))
            phase_poly(qc, co, par[method][0])
            for (j, k) in v['pairs']:
                pair_swap_mixer(qc, 2 * j, 2 * j + 1, 2 * k, 2 * k + 1, par[method][1])
    elif method == 'uniform':
        qc.h(range(NB_Q))
    else:
        biased = method.endswith('_b'); base = method[:-2] if biased else method
        p = par[method]; h, J = baseline_qubo(b, method, p)
        for k, pk in enumerate(warm_p(b, biased)): qc.ry(2 * math.asin(math.sqrt(pk)), k)
        phase_poly(qc, ising_from_qubo(h, J), 1.0)
        if base == 'xy':
            for (a_, b_) in XY_PAIRS: qc.rxx(p[1], a_, b_); qc.ryy(p[1], a_, b_)
        else:
            qc.rx(2 * p[1], range(NB_Q))
    qc.measure(range(NB_Q), range(NB_Q))
    return qc


def probs_from_circuit(qc):
    """exact output distribution in S_ALL order (for tests)"""
    from qiskit.quantum_info import Statevector
    c = qc.remove_final_measurements(inplace=False)
    pv = Statevector(c).probabilities()                                 # little endian: index = sum q_k 2^k
    idx = (S_ALL.astype(int) * (1 << np.arange(NB_Q))).sum(1)
    return pv[idx]


# ----------------------------------------------------------------------------------------------------------------- chip: regions, compile
def edge_error(target, a, b):
    for key in ((a, b), (b, a)):
        try:
            return target['cz'][key].error or 0.0
        except Exception:
            pass
    return 1.0


def device_graph(backend, max_err=0.05):
    """coupling graph without broken couplers (error > max_err) and without unusable qubits (readout error > 0.2)"""
    tg = backend.target; cm = backend.coupling_map
    bad_q = set()
    for q in range(backend.num_qubits):
        try:
            if (tg['measure'][(q,)].error or 0) > 0.2: bad_q.add(q)
        except Exception:
            pass
    edges = {tuple(sorted(e)) for e in cm.get_edges()
             if edge_error(tg, *e) <= max_err and e[0] not in bad_q and e[1] not in bad_q}
    adj = defaultdict(set)
    for a, b in edges: adj[a].add(b); adj[b].add(a)
    return sorted(adj), adj, sorted(edges)


def region_score(backend, reg, adj):
    tg = backend.target; sc = 0.0
    for a in reg:
        for b in adj[a]:
            if b in reg and a < b: sc += edge_error(tg, a, b)
        try:
            sc += 0.5 * (tg['measure'][(a,)].error or 0.0)
        except Exception:
            pass
    return sc


def pack_regions(backend, nblocks, size=11, tries=3000, seed=0):
    nodes, adj, _ = device_graph(backend); rng = np.random.default_rng(seed); best = None
    for _ in range(tries):
        used = set(); regs = []; ok = True
        for b in range(nblocks):
            free = [v for v in nodes if v not in used]
            if len(free) < size: ok = False; break
            fd = {v: sum(u not in used for u in adj[v]) for v in free}
            mn = min(fd.values()); cand0 = [v for v in free if fd[v] <= mn + 1]
            reg = [int(rng.choice(cand0))]
            while len(reg) < size:
                cand = {u for v in reg for u in adj[v] if u not in used and u not in reg}
                if not cand: ok = False; break
                cl = sorted(cand)
                key = [(-sum(w in reg for w in adj[u]), sum(w not in used and w not in reg for w in adj[u]) + (rng.random() < 0.3)) for u in cl]
                m0 = min(key); pool = [u for u, k_ in zip(cl, key) if k_ <= m0]
                reg.append(int(rng.choice(pool)))
            if not ok: break
            regs.append(reg); used.update(reg)
        if not ok: continue
        sc = sum(region_score(backend, set(r), adj) for r in regs)
        if best is None or sc < best[0]: best = (sc, regs)
    if best is None: raise RuntimeError(f'could not pack {nblocks} regions of {size} qubits')
    return best[1]


def sub_esp(t, reg, backend):
    """product of (1 - error) of the gates and readout of a compiled block on physical qubits reg (reg[k] = physical qubit of sub-index k)"""
    tg = backend.target; lg = 0.0; lg_ro = 0.0
    for inst in t.data:
        nm = inst.operation.name
        qs = tuple(reg[t.find_bit(q).index] for q in inst.qubits)
        if nm in ('barrier', 'delay'): continue
        try:
            e = tg[nm][qs].error
        except Exception:
            e = None
        if nm == 'measure': lg_ro += math.log1p(-min(e or 0.0, 0.999))
        elif e: lg += math.log1p(-min(e, 0.999))
    return math.exp(lg), math.exp(lg_ro)


def compile_block(qc, reg, backend, seeds=4):
    """transpile one 10-qubit block onto the connected region reg (physical qubits) using only the region's couplers"""
    _, adj, _ = device_graph(backend); pos = {p: k for k, p in enumerate(reg)}
    edges = [[pos[a], pos[b]] for a in reg for b in adj[a] if b in pos]
    cm = CouplingMap(edges); best = None
    for sd in range(seeds):
        t = transpile(qc, coupling_map=cm, basis_gates=BASIS, optimization_level=3, seed_transpiler=sd + 1)
        esp, esp_ro = sub_esp(t, reg, backend)
        if best is None or esp > best[0]: best = (esp, esp_ro, t)
    esp, esp_ro, t = best
    return t, dict(cz=t.count_ops().get('cz', 0), depth=t.depth(), esp=esp, esp_ro=esp_ro)


def assemble(sub_circuits, regs, n_phys):
    """place the compiled 10-qubit blocks on their physical regions: one circuit of n_phys qubits, 10*nblocks classical bits (clbit 10*b + k = block b, variable k)"""
    nb = len(sub_circuits)
    qr = QuantumRegister(n_phys, 'q'); cr = ClassicalRegister(NB_Q * nb, 'c')
    big = QuantumCircuit(qr, cr)
    for b, (t, reg) in enumerate(zip(sub_circuits, regs)):
        big.compose(t, qubits=list(reg), clbits=list(range(NB_Q * b, NB_Q * (b + 1))), inplace=True)
    return big


def compile_all(inst, params, backend, methods=METHODS, seeds=4, log=print, pack_seed=0, tries=3000, region_size=11):
    """returns {method: dict(circuit=big, blocks=[stats per block], regions=...)}.  Same regions for every method; heavier blocks on better regions."""
    nb = len(inst['blocks']); _, adj, _ = device_graph(backend)
    regs = pack_regions(backend, nb, size=region_size, tries=tries, seed=pack_seed)
    regs = sorted(regs, key=lambda r: region_score(backend, set(r), adj))        # best region first
    out = {}
    for m in methods:
        circs = [block_circuit(b, m, par) for b, par in zip(inst['blocks'], params)]
        # first pass: size of every block (on its own region) to give the heaviest blocks the best regions
        order_cz = []
        for b, qc in enumerate(circs):
            tt = transpile(qc, basis_gates=BASIS, optimization_level=1, seed_transpiler=1)
            order_cz.append(tt.count_ops().get('cz', 0))
        rank = np.argsort(-np.array(order_cz))
        assign = {int(blk): regs[i] for i, blk in enumerate(rank)}
        subs, stats = [], []
        for b, qc in enumerate(circs):
            t, st = compile_block(qc, assign[b], backend, seeds=seeds); subs.append(t); stats.append(dict(st, block=b, region=assign[b]))
        big = assemble(subs, [assign[b] for b in range(nb)], backend.num_qubits)
        out[m] = dict(circuit=big, subs=subs, blocks=stats, regions=[assign[b] for b in range(nb)])
        log(f"{m:11s}: CZ/block median {np.median([s['cz'] for s in stats]):5.0f} (max {max(s['cz'] for s in stats)}), depth/block max {max(s['depth'] for s in stats)}, "
            f"ESP/block median {np.median([s['esp'] for s in stats]):.3f} (min {min(s['esp'] for s in stats):.3f}), total CZ {sum(s['cz'] for s in stats)}")
    return out


# ----------------------------------------------------------------------------------------------------------------- noise model, analysis
def model_probs(ideal, esp_blocks, method):
    """global depolarizing model per block: ESP * ideal + (1 - ESP) * uniform over the 2^10 strings"""
    return [e * d[method] + (1 - e) * 2.0 ** -NB_Q for d, e in zip(ideal, esp_blocks)]


def sample_model(ideal, esp_blocks, method, shots, seed=0):
    """synthetic bitarray (shots, 10*nb) from the model, independent blocks (what the chip would give if the model were exact)"""
    rng = np.random.default_rng(seed); P = model_probs(ideal, esp_blocks, method)
    cols = []
    for p in P:
        idx = rng.choice(2 ** NB_Q, size=shots, p=p / p.sum()); cols.append(S_ALL[idx])
    return np.concatenate(cols, axis=1).astype(np.int8)


def counts_to_bits(counts, nbits):
    """qiskit counts dict -> int8 array (shots, nbits) with column k = classical bit k"""
    rows = []
    for key, c in counts.items():
        bits = np.frombuffer(key.replace(' ', '')[::-1].encode(), dtype=np.uint8) - 48
        rows.append(np.tile(bits[:nbits], (c, 1)))
    return np.concatenate(rows).astype(np.int8)


def analyse(inst, bits, rng=None, n_assemble=2000):
    """bits: (shots, 10*nb).  Per-block feasibility and quality; per-shot count of feasible blocks; assembled solutions (per-block post-selection)."""
    rng = rng or np.random.default_rng(0)
    blocks = inst['blocks']; nb = len(blocks); N = bits.shape[0]
    rows = []; accepted = []
    feas_mat = np.zeros((N, nb), bool)
    for b in blocks:
        x = bits[:, NB_Q * b['id']:NB_Q * (b['id'] + 1)]
        L = x[:, 0::2]; Sh = x[:, 1::2]
        ok = (L.sum(1) == b['ls'][0]) & (Sh.sum(1) == b['ls'][1]) & ((L & Sh).sum(1) == 0)
        feas_mat[:, b['id']] = ok
        idx = (x.astype(int) * (1 << (NB_Q - 1 - np.arange(NB_Q)))).sum(1)         # row index in S_ALL (axis 0 = q0 is the MSB)
        cost = b['cv'][idx]
        q = 1 - (cost[ok].mean() - b['cmin']) / (b['cmax'] - b['cmin']) if ok.any() else float('nan')
        rows.append(dict(block=b['id'], feas=float(ok.mean()), n_acc=int(ok.sum()), quality=float(q),
                         p_opt=float((np.abs(cost - b['cmin']) < 1e-9)[ok].sum() / N), uniform_feas=float(b['mask'].mean())))
        accepted.append(x[ok])
    nacc = np.array([len(a) for a in accepted])
    glob = float(feas_mat.all(1).mean())
    m_assembled = int(nacc.min())
    asm_cost = None
    if m_assembled > 0:
        k = min(m_assembled, n_assemble); X = np.zeros((k, inst['nv']))
        for b, a in zip(blocks, accepted):
            sel = rng.permutation(len(a))[:k]; X[:, b['V']] = a[sel]
        Mx = inst['M']; asm_cost = np.einsum('ki,ij,kj->k', X, Mx, X)
    return dict(blocks=rows, mean_feasible_blocks=float(feas_mat.sum(1).mean()), global_feasible_fraction=glob, shots=N,
                assembled=m_assembled, assembled_cost_mean=None if asm_cost is None else float(asm_cost.mean()),
                assembled_cost_best=None if asm_cost is None else float(asm_cost.min()),
                block_feas_mean=float(np.mean([r['feas'] for r in rows])))


def random_feasible_cost(inst, n=2000, seed=0):
    rng = np.random.default_rng(seed); Mx = inst['M']; c = []
    for _ in range(n):
        x = np.zeros(inst['nv'])
        for b in inst['blocks']: x[b['V']] = S_ALL[rng.choice(np.flatnonzero(b['mask']))]
        c.append(x @ Mx @ x)
    return float(np.mean(c)), float(np.min(c))


def exact_block_cost(inst, sweeps=4, seed=0):
    """classical reference: block coordinate descent with the exact best string of every block (what a perfect block solver reaches)"""
    rng = np.random.default_rng(seed); Mx = inst['M']; nv = inst['nv']; best = None
    for r in range(20):
        x = np.zeros(nv)
        for b in inst['blocks']: x[b['V']] = S_ALL[rng.choice(np.flatnonzero(b['mask']))]
        for _ in range(sweeps):
            for b in inst['blocks']:
                V = b['V']; R = [k for k in range(nv) if k not in set(V)]
                S = S_ALL[b['mask']].astype(float)
                cv = np.einsum('ij,jk,ik->i', S, Mx[np.ix_(V, V)], S) + S @ (2 * Mx[np.ix_(V, R)] @ x[R])
                x[V] = S[np.argmin(cv)]
        c = float(x @ Mx @ x)
        if best is None or c < best: best = c
    return best


# ----------------------------------------------------------------------------------------------------------------- IBM access (key from a local file)
def connect(apikey_path, backend_name='ibm_fez', instance=None):
    from qiskit_ibm_runtime import QiskitRuntimeService
    key = json.load(open(apikey_path))
    token = key.get('apikey', key.get('token')) if isinstance(key, dict) else key
    kw = dict(channel='ibm_quantum_platform', token=token)
    if instance: kw['instance'] = instance
    svc = QiskitRuntimeService(**kw)
    return svc, svc.backend(backend_name)


def compact(qc):
    """drop idle qubits (for single-block noisy simulation)"""
    used = sorted({qc.find_bit(q).index for i in qc.data for q in i.qubits})
    mp = {q: k for k, q in enumerate(used)}
    out = QuantumCircuit(len(used), qc.num_clbits)
    for i in qc.data:
        out.append(i.operation, [mp[qc.find_bit(q).index] for q in i.qubits], [qc.find_bit(c).index for c in i.clbits])
    return out, used


def check_isa(qc, backend):
    """True if every gate is in the basis and every 2-qubit gate sits on a coupler of the backend"""
    edges = {tuple(e) for e in backend.coupling_map.get_edges()}
    for inst in qc.data:
        nm = inst.operation.name
        if nm in ('barrier', 'measure'): continue
        qs = tuple(qc.find_bit(q).index for q in inst.qubits)
        if nm not in BASIS + ['id', 'delay']: return False
        if len(qs) == 2 and qs not in edges and qs[::-1] not in edges: return False
    return True


def wilson(k, n, z=1.96):
    if n == 0: return (float('nan'), float('nan'))
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def aer_blocks(inst, subs, regs, backend, shots=500, blocks=None, seed=0, log=print):
    """gate-level noisy simulation (FakeFez-style noise model) of each block separately; blocks are independent circuits on disjoint qubits,
    so concatenating their samples is what the chip would return without crosstalk.  Returns int8 array (shots, 10*nb)."""
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    nm = NoiseModel.from_backend(backend); sim = AerSimulator(noise_model=nm, seed_simulator=seed)
    nb = len(subs); cols = []
    for b in range(nb):
        if blocks is not None and b not in blocks:
            cols.append(np.zeros((shots, NB_Q), np.int8)); continue
        big = QuantumCircuit(backend.num_qubits, NB_Q); big.compose(subs[b], qubits=list(regs[b]), clbits=list(range(NB_Q)), inplace=True)
        cnt = sim.run(big, shots=shots).result().get_counts()
        bt = counts_to_bits(cnt, NB_Q); cols.append(bt[np.random.default_rng(seed + b).permutation(len(bt))][:shots]); log(f'  block {b} simulated')
    return np.concatenate(cols, axis=1)
