"""Stronger quantum baselines on the residual core (ideal dense simulation, then compiled to ibm_fez and the global depolarising model).

The earlier comparison (case_four.py) gave Fourier-LCU and unbalanced a uniform (Hadamard) start.  The Fourier-LCU paper (arXiv:2605.18985) uses a warm-started product state
RY(2 asin sqrt(k/n)) per qubit, whose feasible probability is Theta(1/sqrt n), and replaces the penalty and the XY mixer by single-qubit layers.  Here every baseline gets that warm start:
  lcu_ws   warm start; objective + exclusivity phase + one Rz angle per count constraint (trained) ; shared single-qubit mixer RX(bx) RY(by) (trained)   [single LCU basis circuit]
  unb_ws   warm start; objective + unbalanced penalty (-l1 g + l2 g^2, both directions) + exclusivity penalty ; X mixer  (best of three (l1,l2) pairs)
  xy_ws    warm start; objective + exclusivity phase ; XY mixer inside every count group (the coherent target of the LCU)  (2 CZ per pair)
  ours     support-F start, objective, pair-exchange mixers (F-basis simulation)
p = 1 and 2 layers, many restarts, several instances (objective seeds x interface tuples).  Metrics: feasible share, quality among feasible, P(opt); then ESP of the compiled
circuits (p=1) and the depolarising-model P(opt) per shot.  Environment: CF_* variables select the core as in case_four.py; BS_SEEDS='5,6,7', BS_TUPLES='7,8', BS_RESTARTS, BS_PMAX, BS_TAG.
Writes results/<BS_TAG>.json (one entry per instance)."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')); _os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)
import json, math, itertools, time, warnings; warnings.filterwarnings('ignore')
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import experiments.case_four as C
import experiments.prune_circuit as P
from negsearch.hw_large import ising_from_qubo
from negsearch.depth_study import phase_poly
from experiments.qaoa_realistic import best as compile_best


def allstrings(f): return ((np.arange(2 ** f)[:, None] >> np.arange(f)) & 1)


class Core:
    def __init__(self, seed_obj, tuple_seed):
        _os.environ['CF_TUPLE_SEED'] = str(tuple_seed)
        cp = C.core_problem(seed_obj=seed_obj); self.cp = cp
        self.free = cp['free']; self.col = cp['col']; self.f = f = len(self.free); self.F = cp['F']; self.nF = len(self.F)
        X = allstrings(f); self.X = X
        cF = C.cost_of(self.F, cp['h1'], cp['J1'], self.col); self.cmin, self.cmax = cF.min(), cF.max()
        self.cF = cF
        cA = C.cost_of(X, cp['h1'], cp['J1'], self.col); self.cn = (cA - self.cmin) / (self.cmax - self.cmin)
        self.codeF = (self.F * (1 << np.arange(f))).sum(1)
        self.featF = np.zeros(2 ** f, bool); self.featF[self.codeF] = True
        self.opt = self.featF & (np.abs(cA - self.cmin) < 1e-9)
        self.groups = P.groups_of(cp['bl'], cp['forced'])
        self.G = np.array([sum(X[:, self.col[v]] for v in vs) - t for vs, t in self.groups], float)
        excl = np.zeros(2 ** f); self.excl_pairs = []
        for b in cp['bl']:
            for j in range(b['size']):
                a, c = b['xq'][2 * j], b['xq'][2 * j + 1]
                if a in self.col and c in self.col: excl += X[:, self.col[a]] * X[:, self.col[c]]; self.excl_pairs.append((a, c))
        self.excl = excl
        # warm start probabilities per free variable
        p0 = np.full(f, 0.5)
        for vs, t in self.groups:
            for v in vs: p0[self.col[v]] = t / len(vs)
        self.p0 = p0
        # XY pairs: path inside each group (ring when > 2 qubits)
        self.xy_pairs = []
        for vs, t in self.groups:
            ks = [self.col[v] for v in vs]
            for i in range(len(ks) - 1): self.xy_pairs.append((ks[i], ks[i + 1]))
            if len(ks) > 2: self.xy_pairs.append((ks[-1], ks[0]))
        self.groups_k = [([self.col[v] for v in vs], t) for vs, t in self.groups]

    def metrics(self, Pr):
        fe = float(Pr[self.featF].sum()); q = float((Pr * self.featF * (1 - self.cn)).sum())
        return dict(feas=fe, quality_feasible=(q / fe if fe > 0 else 0.0), popt=float(Pr[self.opt].sum()))


def warm_state(core):
    f = core.f; vecs = [np.array([math.sqrt(1 - p), math.sqrt(p)], complex) for p in core.p0]      # qubit k = bit k of the index
    psi = vecs[-1]
    for v in reversed(vecs[:-1]): psi = np.kron(psi, v)
    return psi


def axis_of(f, k): return f - 1 - k          # bit k <-> tensor axis f-1-k


def rot_layer(psi, f, gate):
    """gate = 2x2 matrix applied to every qubit"""
    t = psi.reshape([2] * f)
    for ax in range(f):
        a0 = np.take(t, 0, axis=ax); a1 = np.take(t, 1, axis=ax)
        t = np.stack([gate[0, 0] * a0 + gate[0, 1] * a1, gate[1, 0] * a0 + gate[1, 1] * a1], axis=ax)
    return t.reshape(-1)


def RX(b): return np.array([[math.cos(b), -1j * math.sin(b)], [-1j * math.sin(b), math.cos(b)]])
def RY(b): return np.array([[math.cos(b), -math.sin(b)], [math.sin(b), math.cos(b)]])


def xy_layer(psi, f, pairs, beta):
    """exp(-i beta (XX+YY)/2) on every pair, sequentially"""
    t = psi.reshape([2] * f).copy(); c, s = math.cos(beta), -1j * math.sin(beta)
    for (a, b) in pairs:
        ia, ib = axis_of(f, a), axis_of(f, b)
        i01 = [slice(None)] * f; i10 = [slice(None)] * f
        i01[ia], i01[ib] = 0, 1; i10[ia], i10[ib] = 1, 0
        i01, i10 = tuple(i01), tuple(i10)
        x01, x10 = t[i01].copy(), t[i10].copy()
        t[i01] = c * x01 + s * x10; t[i10] = c * x10 + s * x01
    return t.reshape(-1)


# ------------------------------------------------------------------ parameter layouts and states
def n_par(method, core, p):
    G = len(core.G)
    return p * {'lcu_ws': 2 + G + 2, 'unb_ws': 3, 'xy_ws': 3}[method]


def state(method, core, par, p, lam=None):
    f = core.f; psi = warm_state(core); k = 0
    for layer in range(p):
        if method == 'lcu_ws':
            g, lx = par[k], par[k + 1]; th = par[k + 2:k + 2 + len(core.G)]; bx, by = par[k + 2 + len(core.G):k + 4 + len(core.G)]; k += 4 + len(core.G)
            psi = psi * np.exp(-1j * g * core.cn) * np.exp(-1j * lx * core.excl) * np.exp(1j * (th @ core.G))
            psi = rot_layer(psi, f, RX(bx)); psi = rot_layer(psi, f, RY(by))
        elif method == 'unb_ws':
            g, sc, bt = par[k:k + 3]; k += 3; l1, l2 = lam
            pen = np.zeros(2 ** f)
            for row in core.G:
                for sgn in (1, -1): gg = sgn * row; pen += -l1 * gg + l2 * gg ** 2
            psi = psi * np.exp(-1j * (g * core.cn + sc * (pen + 5 * core.excl)))
            psi = rot_layer(psi, f, RX(bt))
        else:
            g, lx, bt = par[k:k + 3]; k += 3
            psi = psi * np.exp(-1j * g * core.cn) * np.exp(-1j * lx * core.excl)
            psi = xy_layer(psi, f, core.xy_pairs, bt)
    return psi


def train(method, core, p, restarts, rng, lam=None, maxiter=60):
    npar = n_par(method, core, p); best = None
    def score(x):
        mt = core.metrics(np.abs(state(method, core, x, p, lam)) ** 2); return -(mt['feas'] * mt['quality_feasible'])
    for r in range(restarts):
        x0 = rng.uniform(-1.5, 1.5, npar)
        o = minimize(score, x0, method='L-BFGS-B', options=dict(maxiter=maxiter))
        if best is None or o.fun < best.fun: best = o
    return best.x, -best.fun


def ours_state(core, p, restarts, rng):
    nF = core.nF; Fm = core.F; idx_of = {tuple(r): i for i, r in enumerate(Fm)}; pairs = []
    for b in core.cp['bl']:
        if len(b['F']) == 1: continue
        for j in range(b['size'] - 1):
            kk = j + 1; cols = [(core.col[b['xq'][2 * j + d]], core.col[b['xq'][2 * kk + d]]) for d in (0, 1) if b['xq'][2 * j + d] in core.col]
            perm = np.empty(nF, int)
            for i, r in enumerate(Fm):
                r2 = r.copy()
                for a_, c_ in cols: r2[a_], r2[c_] = r[c_], r[a_]
                perm[i] = idx_of[tuple(r2)]
            pairs.append(perm)
    def st(x):
        psi = np.ones(nF, complex) / math.sqrt(nF)
        for l in range(p):
            psi = psi * np.exp(-1j * x[2 * l] * core.cF)
            for perm in pairs: psi = math.cos(x[2 * l + 1]) * psi - 1j * math.sin(x[2 * l + 1]) * psi[perm]
        return psi
    best = None
    for r in range(restarts):
        x0 = np.ravel([[rng.uniform(-2, 2), rng.uniform(0, 1.6)] for _ in range(p)])
        o = minimize(lambda x: float((np.abs(st(x)) ** 2 * core.cF).sum()), x0, method='Nelder-Mead', options=dict(maxiter=400))
        if best is None or o.fun < best.fun: best = o
    Pr = np.zeros(2 ** core.f); Pr[core.codeF] = np.abs(st(best.x)) ** 2
    return Pr, best.x


# ------------------------------------------------------------------ circuits (p = 1)
def circuit(method, core, par, lam=None):
    cp = core.cp; nq = len(cp['h1'])
    qc = QuantumCircuit(nq); h, J = cp['h1'].copy(), dict(cp['J1'])
    for v in core.free: qc.ry(2 * math.asin(math.sqrt(core.p0[core.col[v]])), v)
    scale_obj = 1.0 / (core.cmax - core.cmin)             # cn = (c - cmin)/(cmax - cmin): the circuit uses the same normalisation
    h = {v: h[v] * scale_obj for v in range(len(h))}; J = {k: w * scale_obj for k, w in J.items()}
    if method == 'lcu_ws':
        g, lx = par[0], par[1]; th = par[2:2 + len(core.G)]; bx, by = par[2 + len(core.G):4 + len(core.G)]
        hh = np.zeros(len(cp['h1'])); JJ = {}
        for v, w in h.items(): hh[v] += g * w
        for k, w in J.items(): JJ[k] = JJ.get(k, 0) + g * w
        for (a, c) in core.excl_pairs: JJ[(a, c)] = JJ.get((a, c), 0) + lx
        phase_poly(qc, ising_from_qubo(hh, JJ), 1.0)
        for gi, (vs, t) in enumerate(core.groups):
            for v in vs: qc.rz(th[gi], v)           # diag(1, e^{i th}) up to a global phase
        for v in core.free: qc.rx(2 * bx, v); qc.ry(2 * by, v)
    elif method == 'unb_ws':
        g, sc, bt = par[:3]; l1, l2 = lam
        hh = np.zeros(len(cp['h1'])); JJ = {}
        for v, w in h.items(): hh[v] += g * w
        for k, w in J.items(): JJ[k] = JJ.get(k, 0) + g * w
        for (a, c) in core.excl_pairs: JJ[(a, c)] = JJ.get((a, c), 0) + 5 * sc
        for vs, t in core.groups:
            for sgn in (1, -1):
                # sgn*(sum x - t): -l1*sgn*(sum x - t) + l2*(sum x - t)^2
                for u in vs:
                    hh[u] += sc * (-l1 * sgn + l2 * (1 - 2 * t))
                    for w2 in vs:
                        if u < w2: JJ[(u, w2)] = JJ.get((u, w2), 0) + sc * 2 * l2
        phase_poly(qc, ising_from_qubo(hh, JJ), 1.0)
        for v in core.free: qc.rx(2 * bt, v)
    else:
        g, lx, bt = par[:3]
        hh = np.zeros(len(cp['h1'])); JJ = {}
        for v, w in h.items(): hh[v] += g * w
        for k, w in J.items(): JJ[k] = JJ.get(k, 0) + g * w
        for (a, c) in core.excl_pairs: JJ[(a, c)] = JJ.get((a, c), 0) + lx
        phase_poly(qc, ising_from_qubo(hh, JJ), 1.0)
        for (a, b) in core.xy_pairs: qc.rxx(bt, core.free[a], core.free[b]); qc.ryy(bt, core.free[a], core.free[b])
    return qc


def circuit_probs(core, qc):
    """probabilities over the dense core space from the compiled-circuit statevector (compacted to the free qubits)"""
    c, used = C.compact(qc); sv = Statevector(c).probabilities(); loc = {v: k for k, v in enumerate(used)}
    codes = np.zeros(len(sv), int)
    for s_i in range(len(sv)):
        code = 0
        for v in core.free:
            if (s_i >> loc[v]) & 1: code |= 1 << core.col[v]
        codes[s_i] = code
    out = np.zeros(2 ** core.f); np.add.at(out, codes, sv); return out


def run_instance(seed_obj, tuple_seed, R, PMAX, check=True):
    t0 = time.time(); core = Core(seed_obj, tuple_seed); rng = np.random.default_rng(seed_obj * 100 + tuple_seed)
    out = dict(seed_obj=seed_obj, tuple_seed=tuple_seed, n=C.n, free=core.f, nF=core.nF, iface=[[list(c) for c in p] for p in core.cp['iface']],
               groups=len(core.groups), methods={})
    print(f"instance obj={seed_obj} tuple={tuple_seed}: free={core.f} |F|={core.nF} groups={len(core.groups)}", flush=True)
    Pu = np.zeros(2 ** core.f); Pu[core.codeF] = 1 / core.nF; out['uniform_on_F'] = core.metrics(Pu)
    for p in range(1, PMAX + 1):
        Pq, xq = ours_state(core, p, R, rng); out['methods'][f'ours_p{p}'] = dict(ideal=core.metrics(Pq), par=[float(v) for v in xq])
        for m in ('lcu_ws', 'unb_ws', 'xy_ws'):
            lams = [(0.96, 0.037), (0.5, 0.2), (2.0, 0.5)] if m == 'unb_ws' else [None]
            bestm = None
            for lam in lams:
                x, sc = train(m, core, p, R, rng, lam)
                if bestm is None or sc > bestm[1]: bestm = (x, sc, lam)
            x, sc, lam = bestm; Pm = np.abs(state(m, core, x, p, lam)) ** 2
            out['methods'][f'{m}_p{p}'] = dict(ideal=core.metrics(Pm), par=[float(v) for v in x], lam=lam)
            print(f"  p={p} {m:7s}: feas {out['methods'][f'{m}_p{p}']['ideal']['feas']:.3f} q {out['methods'][f'{m}_p{p}']['ideal']['quality_feasible']:.3f} popt {out['methods'][f'{m}_p{p}']['ideal']['popt']:.2e}  ({time.time()-t0:.0f}s)", flush=True)
    # compile p=1 circuits (ours: case_four circuit with the p=1 angles) and the depolarising model
    xq = np.array(out['methods']['ours_p1']['par'])
    qc_q = P.circuit(C.A, C.T, C.S, core.cp['bl'], core.cp['h1'], core.cp['J1'], [xq[0]], [xq[1]], True)
    circs = {'ours': qc_q}
    for m in ('lcu_ws', 'unb_ws', 'xy_ws'):
        d = out['methods'][f'{m}_p1']; circs[m] = circuit(m, core, np.array(d['par']), d['lam'])
    if check and core.f <= 16:
        for m in ('lcu_ws', 'xy_ws', 'unb_ws'):
            d = out['methods'][f'{m}_p1']
            Pc = circuit_probs(core, circs[m]); Pm = np.abs(state(m, core, np.array(d['par']), 1, d['lam'])) ** 2
            out['methods'][f'{m}_p1']['circuit_vs_ideal_tv'] = float(0.5 * np.abs(Pc - Pm).sum())
            print(f"  circuit check {m}: TV {out['methods'][f'{m}_p1']['circuit_vs_ideal_tv']:.2e}", flush=True)
    for k, qc in circs.items():
        r = compile_best(qc, seeds=4); key = 'ours_p1' if k == 'ours' else f'{k}_p1'; out['methods'][key]['compiled'] = r
        print(f"  compiled {k:7s}: CZ {r['cz']} depth {r['depth']} ESP {r['esp']:.3g}", flush=True)
    # depolarising model:  ESP * ideal + (1 - ESP) * uniform
    for key in [k for k in out['methods'] if k.endswith('_p1')]:
        d = out['methods'][key]; e = d['compiled']['esp']
        if key == 'ours_p1': Pm = ours_fixed(core, xq)
        else: Pm = np.abs(state(key[:-3], core, np.array(d['par']), 1, d['lam'])) ** 2
        d['model'] = core.metrics(e * Pm + (1 - e) / 2 ** core.f)
    out['seconds'] = time.time() - t0
    return out


def ours_fixed(core, x):
    nF = core.nF; Fm = core.F; idx_of = {tuple(r): i for i, r in enumerate(Fm)}; pairs = []
    for b in core.cp['bl']:
        if len(b['F']) == 1: continue
        for j in range(b['size'] - 1):
            kk = j + 1; cols = [(core.col[b['xq'][2 * j + d]], core.col[b['xq'][2 * kk + d]]) for d in (0, 1) if b['xq'][2 * j + d] in core.col]
            perm = np.empty(nF, int)
            for i, r in enumerate(Fm):
                r2 = r.copy()
                for a_, c_ in cols: r2[a_], r2[c_] = r[c_], r[a_]
                perm[i] = idx_of[tuple(r2)]
            pairs.append(perm)
    psi = np.ones(nF, complex) / math.sqrt(nF); psi = psi * np.exp(-1j * x[0] * core.cF)
    for perm in pairs: psi = math.cos(x[1]) * psi - 1j * math.sin(x[1]) * psi[perm]
    Pr = np.zeros(2 ** core.f); Pr[core.codeF] = np.abs(psi) ** 2; return Pr


if __name__ == '__main__':
    seeds = [int(v) for v in _os.environ.get('BS_SEEDS', '5').split(',')]; tuples = [int(v) for v in _os.environ.get('BS_TUPLES', '7').split(',')]
    R = int(_os.environ.get('BS_RESTARTS', 12)); PMAX = int(_os.environ.get('BS_PMAX', 2)); TAG = _os.environ.get('BS_TAG', 'baselines_strong')
    allres = []
    for so in seeds:
        for tu in tuples:
            try:
                allres.append(run_instance(so, tu, R, PMAX))
            except Exception as e:
                import traceback; traceback.print_exc(); print('instance failed', so, tu, e, flush=True)
            json.dump(allres, open(f'results/{TAG}.json', 'w'), indent=1, default=float)
