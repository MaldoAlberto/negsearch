"""Larger case (core of 24-36 free qubits).  Dense simulation of the baselines is impossible at this size, so:
  ours-QAOA  : ideal simulation inside F (exact, |F| <= ~1e6), angles optimised; compiled circuit CZ/depth/ESP on ibm_fez
  Fourier-LCU / unbalanced : compiled CZ/depth/ESP only (angles do not change the gate count); ideal feasibility NOT simulated
  ours-Grover: iterations needed (from |F|), CZ of one iteration (compiled, ancilla-free reflection)
Env: CF_M, CF_T, CF_FAM, CF_FMIN, CF_FMAX, CF_LOGFMAX, CF_TAG.  Writes results/<CF_TAG>.json."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)
import json, math, os, warnings; warnings.filterwarnings('ignore')
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
import experiments.case_four as C
import experiments.prune_circuit as P
from negsearch.hw_large import ising_from_qubo
from negsearch.depth_study import phase_poly
from negsearch.symlower import symmetric_block_prep
from experiments.qaoa_realistic import best as compile_best

cp = C.core_problem(); free, col, Fm, bl = cp['free'], cp['col'], cp['F'], cp['bl']; f = len(free); nF = len(Fm); n = C.n
print(f"n={n} variables, free qubits f={f}, |F_core|={nF} (log2 {math.log2(nF):.1f})", flush=True)
idx_of = {tuple(r): i for i, r in enumerate(Fm)}
cF = C.cost_of(Fm, cp['h1'], cp['J1'], col); cmin, cmax = cF.min(), cF.max(); cn = (cF - cmin) / (cmax - cmin); opt = np.abs(cF - cmin) < 1e-9
pairs = []
for b in bl:
    if len(b['F']) == 1: continue
    for j in range(b['size'] - 1):
        k = j + 1; cols = [(col[b['xq'][2 * j + d]], col[b['xq'][2 * k + d]]) for d in (0, 1) if b['xq'][2 * j + d] in col]
        perm = np.empty(nF, int)
        for i, r in enumerate(Fm):
            r2 = r.copy()
            for a_, c_ in cols: r2[a_], r2[c_] = r[c_], r[a_]
            perm[i] = idx_of[tuple(r2)]
        pairs.append(perm)
def st(g, bt):
    psi = np.ones(nF, complex) / math.sqrt(nF) * np.exp(-1j * g * cF)
    for perm in pairs: psi = math.cos(bt) * psi - 1j * math.sin(bt) * psi[perm]
    return psi
rng = np.random.default_rng(0); bestp = None
for r in range(20):
    o = minimize(lambda x: float((np.abs(st(*x)) ** 2 * cF).sum()), [rng.uniform(-2, 2), rng.uniform(0, 1.6)], method='Nelder-Mead')
    if bestp is None or o.fun < bestp.fun: bestp = o
g_, b_ = bestp.x; Pf = np.abs(st(g_, b_)) ** 2
ideal = dict(feas=1.0, quality_feasible=float((Pf * (1 - cn)).sum()), popt=float(Pf[opt].sum()))
uni = dict(quality_feasible=float((1 - cn).mean()), popt=float(opt.mean()))
out = dict(n=n, free=f, nF=nF, log2F=math.log2(nF), iface=[[list(c) for c in p] for p in cp['iface']], ours_qaoa=dict(ideal=ideal), uniform_on_F=uni)
qc_q = P.circuit(C.A, C.T, C.S, bl, cp['h1'], cp['J1'], [g_], [b_], True)
groups = P.groups_of(bl, cp['forced'])
def base(extra_h=None, extra_J=None, rz=False):
    h, J = cp['h1'].copy(), dict(cp['J1'])
    for v, w in (extra_h or {}).items(): h[v] += w
    for k, w in (extra_J or {}).items(): J[k] = J.get(k, 0) + w
    qc = QuantumCircuit(n)
    for v in free: qc.h(v)
    phase_poly(qc, ising_from_qubo(h, J), 0.3)
    if rz:
        for vs, tgt in groups:
            for v in vs: qc.rz(0.4, v)
    for v in free: qc.rx(0.8, v)
    return qc
exJ = {}; penJ = {}; penh = {}
for b in bl:
    for j in range(b['size']):
        a, c = b['xq'][2 * j], b['xq'][2 * j + 1]
        if a in col and c in col: exJ[(a, c)] = 0.1; penJ[(a, c)] = 0.1
for vs, tgt in groups:
    for i, u in enumerate(vs):
        penh[u] = penh.get(u, 0) + 0.1
        for w2 in vs[i + 1:]: k = (min(u, w2), max(u, w2)); penJ[k] = penJ.get(k, 0) + 0.1
prep = QuantumCircuit(n)
for b in bl: prep.compose(symmetric_block_prep(b['size'], b['l'], b['s']), qubits=b['xq'], inplace=True)
gc, _ = C.compact(prep); fq = gc.num_qubits
it = QuantumCircuit(fq); it.compose(gc.inverse(), inplace=True); it.x(range(fq)); it.h(fq - 1); it.mcx(list(range(fq - 1)), fq - 1); it.h(fq - 1); it.x(range(fq)); it.compose(gc, inplace=True)
for k, qc in [('ours_qaoa', qc_q), ('lcu', base(extra_J=exJ, rz=True)), ('unbalanced', base(extra_h=penh, extra_J=penJ)), ('ours_grover_iter', it)]:
    r = compile_best(C.compact(qc)[0], seeds=3); out.setdefault(k, {})['compiled'] = r
    print(f"{k:18s} CZ {r['cz']:6d} depth {r['depth']:6d} qubits {r['qubits']:3d} ESP {r['esp']:.3g}", flush=True)
e = out['ours_qaoa']['compiled']['esp']
out['ours_qaoa']['noisy_model'] = dict(feas=e + (1 - e) * nF / 2 ** f, popt=e * ideal['popt'] + (1 - e) / 2 ** f * opt.sum())
k_opt = max(1, int(round(math.pi / 4 * math.sqrt(nF / max(1, opt.sum()))))); out['ours_grover'] = dict(iters=k_opt, iters_std_over_ours_log2=(f - math.log2(nF)) / 2,
    total_cz=k_opt * out['ours_grover_iter']['compiled']['cz'])
print(json.dumps(dict(ideal=ideal, uniform=uni, noisy=out['ours_qaoa']['noisy_model'], grover=out['ours_grover']), indent=1, default=float))
json.dump(out, open(f"results/{_os.environ.get('CF_TAG', 'case_large')}.json", 'w'), indent=1, default=float)
